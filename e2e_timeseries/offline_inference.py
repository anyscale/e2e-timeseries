"""
# Offline Inference with DLinear and Ray Data

This tutorial demonstrates how to perform batch inference using the DLinear model and Ray Data.
We load the model checkpoint, prepare the test data, run inference in batches, and evaluate the performance.
"""

# %% [code] - Imports and Environment Setup
import argparse
import os

import numpy as np
import ray
import torch

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

from e2e_timeseries.data_provider.data_factory import data_provider
from e2e_timeseries.models import DLinear
from e2e_timeseries.utils.metrics import metric

"""
The above cell imports the necessary libraries and sets up the environment for Ray Data.
It also imports the data provider, model, and evaluation metric used in the inference pipeline.
"""


# %% [code] - Predictor Class Definition
class Predictor:
    """Actor class for performing inference with the DLinear model."""

    def __init__(self, checkpoint_path: str, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model from checkpoint
        self.model = DLinear.Model(config).float()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict:
        """Process a batch of data for inference (numpy batch format)."""
        # Convert input batch to tensor
        batch_x = torch.from_numpy(batch["x"]).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_x)  # Shape (N, pred_len, features_out)

        # Determine feature dimension based on config
        f_dim = -1 if self.config["features"] == "MS" else 0
        outputs = outputs[:, -self.config["pred_len"] :, f_dim:]
        outputs_np = outputs.cpu().numpy()

        # Extract the target part from the batch
        batch_y = batch["y"]
        batch_y_target = batch_y[:, -self.config["pred_len"] :]

        return {"predictions": outputs_np, "targets": batch_y_target}


"""
This cell defines the Predictor class. It loads the trained DLinear model from a checkpoint and
processes input batches to produce predictions. The __call__ method is used to perform inference
on a given batch of numpy arrays.
"""

# %% [code] - Argument Parsing Setup


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Inference for DLinear model on ETTh1")

    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file (.pt)")

    # Data loader arguments (should match training configuration)
    parser.add_argument("--root_path", type=str, default="./e2e_timeseries/dataset/", help="Root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="Data file name")
    parser.add_argument("--num_data_workers", type=int, default=1, help="Number of workers for PyTorch DataLoader during inference")
    parser.add_argument("--features", type=str, default="S", help="Forecasting task type (M, S, MS)")
    parser.add_argument("--target", type=str, default="OT", help="Target feature in S or MS task")
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run a smoke test")

    # Model configuration arguments
    parser.add_argument("--seq_len", type=int, default=96, help="Input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="Start token length (not used by DLinear but part of dataset structure)")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction sequence length")
    parser.add_argument("--individual", action="store_true", default=False, help="DLinear: individual layers per channel")
    parser.add_argument("--enc_in", type=int, default=1, help="Encoder input size (set based on features)")  # Default for 'S'

    # Inference configuration
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--num_predictor_replicas", type=int, default=1, help="Number of Predictor replicas")

    args = parser.parse_args()

    # Configure encoder input size based on task type
    if args.features == "M" or args.features == "MS":
        args.enc_in = 7  # ETTh1 has 7 features when multiple features are used
    else:
        args.enc_in = 1

    # Ensure paths are absolute
    args.root_path = os.path.abspath(args.root_path)
    args.data_path = os.path.abspath(os.path.join(args.root_path, args.data_path))
    args.checkpoint_path = os.path.abspath(args.checkpoint_path)

    if torch.cuda.is_available():
        print("CUDA is available, using GPU and setting num_gpus_per_worker to 1.0")
        args.num_gpus_per_worker = 1.0
    else:
        print("CUDA is not available, using CPU and setting num_gpus_per_worker to 0.0")
        args.num_gpus_per_worker = 0.0

    return vars(args)


"""
This cell parses command-line arguments needed for the inference,
sets up the model configuration, and converts file paths to absolute paths.
It also determines if the GPU should be used.
"""

# %% [code] - Main Inference Pipeline


def main():
    config = parse_args()
    config["train_only"] = False  # load test subset

    ray.init(ignore_reinit_error=True)

    print("Loading test data...")
    ds = data_provider(config, flag="test")

    ds = ds.map_batches(
        Predictor,
        fn_constructor_kwargs={"checkpoint_path": config["checkpoint_path"], "config": config},
        batch_size=config["batch_size"],
        concurrency=config["num_predictor_replicas"],
        num_gpus=config["num_gpus_per_worker"],
        batch_format="numpy",
    )

    def postprocess_items(item: dict) -> dict:
        # Squeeze singleton dimensions for predictions and targets if necessary
        if item["predictions"].shape[-1] == 1:
            item["predictions"] = item["predictions"].squeeze(-1)
        if item["targets"].shape[-1] == 1:
            item["targets"] = item["targets"].squeeze(-1)
        return item

    ds = ds.map(postprocess_items)

    # Trigger the lazy execution of the Ray pipeline
    all_results = ds.take_all()

    # Concatenate predictions and targets from all batches
    all_predictions = np.concatenate([item["predictions"] for item in all_results], axis=0)
    all_targets = np.concatenate([item["targets"] for item in all_results], axis=0)

    # Compute evaluation metrics
    mae, mse, rmse, mape, mspe, rse = metric(all_predictions, all_targets)

    print("\n--- Test Results ---")
    print(f"MSE: {mse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.3f}")
    print(f"MSPE: {mspe:.3f}")
    print(f"RSE: {rse:.3f}")

    print("\nOffline inference finished!")


"""
This cell defines the main function which ties together the entire inference pipeline:
- It parses configuration parameters
- Initializes Ray
- Loads test data using the data provider
- Applies the Predictor for batch inference
- Post-processes the results and computes evaluation metrics
"""

# %% [code] - Execution
if __name__ == "__main__":
    main()

"""
This final cell triggers the inference pipeline when the script is executed directly.
"""
