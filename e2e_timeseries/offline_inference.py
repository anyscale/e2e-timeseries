"""
Offline inference script for DLinear model on ETT dataset using Ray Data.
"""
import argparse
import os

import numpy as np
import ray
import torch

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

from data_provider.data_factory import data_provider
from models import DLinear
from utils.metrics import metric


class Predictor:
    """Actor class for performing inference with the DLinear model."""

    def __init__(self, checkpoint_path: str, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model from checkpoint
        self.model = DLinear.Model(args).float()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # raise Exception(f"checkpoint keys: {checkpoint.keys()}")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    def __call__(self, batch: dict) -> dict:
        """Process a batch of data for inference."""
        # Ensure input is a dictionary of numpy arrays as expected by map_batches
        batch_x = torch.from_numpy(batch["x"]).float().to(self.device)
        

        with torch.no_grad():
            outputs = self.model(batch_x)

        # Extract predictions and targets based on model config
        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.args.pred_len :, f_dim:]
        batch_y = batch["y"]
        batch_y_target = batch_y[:, -self.args.pred_len :, f_dim:]

        # Return numpy arrays
        return {"predictions": outputs.cpu().numpy(), "targets": batch_y_target}


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Inference for DLinear model on ETTh1")

    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file (.pt)")

    # Data loader args (should match the training configuration)
    parser.add_argument("--root_path", type=str, default="./e2e_timeseries/dataset/", help="Root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="Data file name")
    parser.add_argument("--num_data_workers", type=int, default=1, help="Number of workers for PyTorch DataLoader during inference") # Usually fewer needed for inference
    parser.add_argument("--features", type=str, default="S", help="Forecasting task type (M, S, MS)")
    parser.add_argument("--target", type=str, default="OT", help="Target feature in S or MS task")
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run a smoke test")

    # Model configuration args (must match the trained model)
    parser.add_argument("--seq_len", type=int, default=96, help="Input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="Start token length (not used by DLinear but part of dataset structure)")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction sequence length")
    parser.add_argument("--individual", action="store_true", default=False, help="DLinear: individual layers per channel")
    parser.add_argument("--enc_in", type=int, default=1, help="Encoder input size (set based on features)") # Default for 'S'

    # Inference config
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use GPU for inference if available")
    parser.add_argument("--num_gpus_per_worker", type=float, default=0.0, help="Number of GPUs to assign to each Predictor worker")
    parser.add_argument("--num_predictor_replicas", type=int, default=1, help="Number of Predictor replicas")

    args = parser.parse_args()

    # Determine enc_in based on features
    if args.features == 'M' or args.features == 'MS':
        args.enc_in = 7 # ETTh1 has 7 features
    else: # 'S'
        args.enc_in = 1

    # Ensure paths are absolute
    args.root_path = os.path.abspath(args.root_path)
    args.data_path = os.path.abspath(os.path.join(args.root_path, args.data_path))
    args.checkpoint_path = os.path.abspath(args.checkpoint_path)

    if args.use_gpu and torch.cuda.is_available():
        if args.num_gpus_per_worker == 0.0:
            print("Warning: --use_gpu is set but --num_gpus_per_worker is 0. Defaulting to 1 GPU per worker.")
            args.num_gpus_per_worker = 1.0
    elif args.use_gpu and not torch.cuda.is_available():
        print("Warning: --use_gpu requested but CUDA not available. Using CPU.")
        args.use_gpu = False
        args.num_gpus_per_worker = 0.0
    else: # Not using GPU
        args.num_gpus_per_worker = 0.0




    return args

def main():
    args = parse_args()
    args.train_only = False  # Important: Load test data

    ray.init(ignore_reinit_error=True)

    print("Loading test data...")
    # Use data_provider to get a DataLoader for the 'test' set
    # Note: We set train_only=False and don't need smoke_test here

    _test_loader, ds = data_provider(args, flag='test')

    # Convert DataLoader to Ray Dataset
    # The DataLoader yields tuples (x, y)
    # Ray Data automatically converts this into records like {"item": [x, y]}
    ds = ray.data.from_torch(ds)

    # Preprocess items into the expected input format for the Predictor
    def preprocess_items(item):
        return {"x": np.array(item["item"][0]), "y": np.array(item["item"][1])}
    
    ds = ds.map(preprocess_items)


    raise Exception(f"test_ds: {ds.take(1)}")

    # test_ds.show(1)

    print("Starting inference...")
    # Perform inference using map_batches with the Predictor class
    predictions_ds = ds.map_batches(
        Predictor,
        fn_constructor_kwargs={"checkpoint_path": args.checkpoint_path, "args": args},
        batch_size=args.batch_size, # Process N samples per actor call
        concurrency=args.num_predictor_replicas,
        num_gpus=args.num_gpus_per_worker if args.use_gpu else 0,
        batch_format="pandas"
    )

    print(f"Inference complete. Predictions dataset schema: {predictions_ds.schema()}")
    # predictions_ds.show(1)

    print("Calculating metrics on the full test set...")

    # Collect all predictions and targets from the distributed dataset
    # This might be memory intensive for very large datasets.
    # Consider alternatives like custom aggregation if memory becomes an issue.
    all_results = predictions_ds.take_all()

    if not all_results:
        print("Error: No results were collected from the prediction dataset.")
        return

    # Concatenate predictions and targets from all batches
    all_predictions = np.concatenate([item['predictions'] for item in all_results], axis=0)
    all_targets = np.concatenate([item['targets'] for item in all_results], axis=0)

    print(f"Collected predictions shape: {all_predictions.shape}")
    print(f"Collected targets shape: {all_targets.shape}")

    # Calculate final metrics on the complete dataset
    mae, mse, rmse, mape, mspe, rse = metric(all_predictions, all_targets)

    print("\n--- Test Results ---")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"MSPE: {mspe:.4f}")
    print(f"RSE: {rse:.4f}")

    print("\nOffline inference finished.")

if __name__ == "__main__":
    main()
