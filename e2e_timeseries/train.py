import os
import tempfile

# Enable Ray Train V2
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import random
import time
import warnings

import numpy as np
import ray
import ray.train.torch
import torch
import torch.nn as nn
from ray import train
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch import optim

from e2e_timeseries.data_provider.data_factory import data_provider
from e2e_timeseries.models import DLinear
from e2e_timeseries.utils.metrics import metric
from e2e_timeseries.utils.tools import adjust_learning_rate

warnings.filterwarnings("ignore")


def train_loop_per_worker(config: dict):
    """Main training loop adapted for Ray Train workers."""

    fix_seed = config["fix_seed"] if "fix_seed" in config else 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Automatically determine device based on availability
    device = train.torch.get_device()

    def _get_processed_outputs_and_targets(raw_outputs, batch_y_on_device, config_inner):
        """
        Processes model outputs and batch_y by slicing them to the prediction length
        and selecting the appropriate features based on the task type.
        Assumes batch_y_on_device is already on the correct device.
        """
        pred_len = config_inner["pred_len"]
        f_dim_start_index = -1 if config_inner["features"] == "MS" else 0

        # Slice for prediction length first
        outputs_pred_len = raw_outputs[:, -pred_len:, :]
        batch_y_pred_len = batch_y_on_device[:, -pred_len:, :]

        # Then slice for features
        final_outputs = outputs_pred_len[:, :, f_dim_start_index:]
        final_batch_y_target = batch_y_pred_len[:, :, f_dim_start_index:]

        return final_outputs, final_batch_y_target

    # === Build Model ===
    model = DLinear.Model(config).float()
    model = train.torch.prepare_model(model)
    model.to(device)

    # === Get Data ===
    train_ds = data_provider(config, flag="train")
    if not config["train_only"]:
        val_ds = data_provider(config, flag="val")

    # === Optimizer and Criterion ===
    model_optim = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    # === AMP Scaler ===
    scaler = None
    if config["use_amp"]:
        scaler = torch.amp.GradScaler("cuda")

    # === Training Loop ===
    for epoch in range(config["train_epochs"]):
        model.train()
        train_loss_epoch = []
        epoch_start_time = time.time()

        # Iterate over Ray Dataset batches. The dataset now yields dicts {'x': numpy_array, 'y': numpy_array}
        # iter_torch_batches will convert these to Torch tensors and move to device.
        for batch in train_ds.iter_torch_batches(batch_size=config["batch_size"], device=device, dtypes=torch.float32):
            model_optim.zero_grad()
            batch_x = batch["x"]  # Already a tensor on the correct device
            batch_y = batch["y"]  # Already a tensor on the correct device

            # Forward pass
            if config["use_amp"]:
                with torch.amp.autocast("cuda"):
                    raw_outputs = model(batch_x)
                    outputs, batch_y_target = _get_processed_outputs_and_targets(raw_outputs, batch_y, config)
                    loss = criterion(outputs, batch_y_target)
            else:
                raw_outputs = model(batch_x)
                outputs, batch_y_target = _get_processed_outputs_and_targets(raw_outputs, batch_y, config)
                loss = criterion(outputs, batch_y_target)

            train_loss_epoch.append(loss.item())

            # Backward pass
            if config["use_amp"]:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

        # === End of Epoch ===
        epoch_train_loss = np.average(train_loss_epoch)
        epoch_duration = time.time() - epoch_start_time

        results_dict = {
            "epoch": epoch + 1,
            "train/loss": epoch_train_loss,
            "epoch_duration_s": epoch_duration,
        }

        # === Validation ===
        if not config["train_only"]:
            model.eval()
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for batch in val_ds.iter_torch_batches(batch_size=config["batch_size"], device=device, dtypes=torch.float32):
                    batch_x, batch_y = batch["x"], batch["y"]

                    if config["use_amp"] and torch.cuda.is_available():
                        with torch.amp.autocast("cuda"):
                            raw_outputs = model(batch_x)
                    else:
                        raw_outputs = model(batch_x)

                    outputs, batch_y_target = _get_processed_outputs_and_targets(raw_outputs, batch_y, config)

                    all_preds.append(outputs.detach().cpu().numpy())
                    all_trues.append(batch_y_target.detach().cpu().numpy())

            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)

            mae, mse, rmse, mape, mspe, rse = metric(all_preds, all_trues)

            results_dict["val/loss"] = mse
            results_dict["val/mae"] = mae
            results_dict["val/rmse"] = rmse
            results_dict["val/mape"] = mape
            results_dict["val/mspe"] = mspe
            results_dict["val/rse"] = rse

            print(f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.7f}, Val Loss: {mse:.7f}, Val MSE: {mse:.7f} (Duration: {epoch_duration:.2f}s)")

        # === Reporting and Checkpointing ===
        if train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        "optimizer_state_dict": model_optim.state_dict(),
                        "train_args": config,
                    },
                    os.path.join(temp_checkpoint_dir, "checkpoint.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(metrics=results_dict, checkpoint=checkpoint)
        else:
            train.report(metrics=results_dict, checkpoint=None)

        adjust_learning_rate(model_optim, epoch + 1, config)


if __name__ == "__main__":
    # Define configuration directly
    config = {
        # basic config
        "train_only": False,
        "smoke_test": False,  # Set to True to run a smoke test
        # data loader args
        "root_path": "./e2e_timeseries/dataset/",
        "num_data_workers": 10,
        # forecasting task type
        # S: univariate predict univariate
        # M: multivariate predict univariate
        # MS: multivariate predict multivariate
        "features": "S",
        "target": "OT",  # target variable name for prediction
        "checkpoints": "./checkpoints/",
        # forecasting task args
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 96,
        # DLinear specific args
        "individual": False,
        # optimization args
        "num_replicas": 1,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 3,  # Note: early stopping not implemented in this script
        "learning_rate": 0.005,
        "loss": "mse",
        "lradj": "type1",
        "use_amp": False,
        # Other args
        "fix_seed": 2021,
    }

    # Set dataset specific args
    config["data"] = "ETTh1"
    config["data_path"] = "ETTh1.csv"
    if config["features"] == "S":  # S: univariate predict univariate
        config["enc_in"] = 1
    else:  # M or MS
        config["enc_in"] = 7  # ETTh1 has 7 features

    # Ensure paths are absolute
    config["root_path"] = os.path.abspath(config["root_path"])
    config["data_path"] = os.path.abspath(os.path.join(config["root_path"], config["data_path"]))
    config["checkpoints"] = os.path.abspath(config["checkpoints"])

    # --- Smoke Test Modifications ---
    if config["smoke_test"]:
        print("--- RUNNING SMOKE TEST ---")
        config["train_epochs"] = 2
        config["batch_size"] = 2
        config["num_data_workers"] = 1

    # === Ray Train Setup ===
    ray.init()

    use_gpu = "GPU" in ray.cluster_resources() and ray.cluster_resources()["GPU"] >= 1
    print(f"Using GPU: {use_gpu}")
    scaling_config = ScalingConfig(num_workers=config["num_replicas"], use_gpu=use_gpu, resources_per_worker={"GPU": 1} if use_gpu else None)

    # Adjust run name for smoke test
    run_name_prefix = "SmokeTest_" if config["smoke_test"] else ""
    run_name = f"{run_name_prefix}DLinear_{config['data']}_{config['features']}_{config['target']}_{time.strftime('%Y%m%d_%H%M%S')}"

    run_config = RunConfig(
        storage_path=config["checkpoints"],
        name=run_name,
        checkpoint_config=CheckpointConfig(num_to_keep=2, checkpoint_score_attribute="val/loss", checkpoint_score_order="min"),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # === Run Training ===
    print("Starting Ray Train job...")
    result = trainer.fit()
    print("Training finished.")

    # === Post-Training ===
    if result.best_checkpoints:
        best_checkpoint_path = None
        if not config["train_only"] and "val/loss" in result.metrics_dataframe:
            best_checkpoint = result.get_best_checkpoint(metric="val/loss", mode="min")
            if best_checkpoint:
                best_checkpoint_path = best_checkpoint.path
        elif "train/loss" in result.metrics_dataframe:  # Fallback or if train_only
            best_checkpoint = result.get_best_checkpoint(metric="train/loss", mode="min")
            if best_checkpoint:
                best_checkpoint_path = best_checkpoint.path

        if best_checkpoint_path:
            print("Best checkpoint found:")
            print(f"  Directory: {best_checkpoint_path}")
        else:
            print("Could not retrieve the best checkpoint based on available metrics.")
    else:
        print("No checkpoints were saved during training.")
