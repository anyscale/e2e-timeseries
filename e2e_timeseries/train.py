import os
import tempfile

# Enable Ray Train V2
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import argparse
import random
import time
import warnings

import numpy as np
import ray
import ray.train.torch
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from models import DLinear
from ray import train
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch import optim
from utils.metrics import metric
from utils.tools import adjust_learning_rate

warnings.filterwarnings("ignore")


def train_loop_per_worker(config: dict):
    """Main training loop adapted for Ray Train workers."""
    args = argparse.Namespace(**config)  # Convert dict back to Namespace for compatibility

    fix_seed = args.fix_seed if hasattr(args, "fix_seed") else 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu:
        device = train.torch.get_device()
    else:
        device = torch.device("cpu")

    # === Build Model ===
    model = DLinear.Model(args).float()
    model = train.torch.prepare_model(model)
    model.to(device)

    # === Get Data ===
    train_loader = data_provider(args, flag="train")
    if not args.train_only:
        val_loader = data_provider(args, flag="val")

    train_loader = train.torch.prepare_data_loader(train_loader)
    if not args.train_only:
        val_loader = train.torch.prepare_data_loader(val_loader)

    # === Optimizer and Criterion ===
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # === AMP Scaler ===
    scaler = None
    if args.use_amp:
        scaler = torch.amp.GradScaler("cuda")

    # === Training Loop ===
    for epoch in range(args.train_epochs):
        model.train()
        train_loss_epoch = []
        epoch_start_time = time.time()

        for i, (batch_x, batch_y) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # Forward pass
            if args.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(batch_x)
                    f_dim = -1 if args.features == "MS" else 0
                    outputs = outputs[:, -args.pred_len :, f_dim:]
                    batch_y_target = batch_y[:, -args.pred_len :, f_dim:].to(device)
                    loss = criterion(outputs, batch_y_target)
            else:
                outputs = model(batch_x)
                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len :, f_dim:]
                batch_y_target = batch_y[:, -args.pred_len :, f_dim:].to(device)
                loss = criterion(outputs, batch_y_target)

            train_loss_epoch.append(loss.item())

            # Backward pass
            if args.use_amp:
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
        if not args.train_only:
            model.eval()
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)

                    if args.use_amp:
                        with torch.amp.autocast("cuda"):
                            outputs = model(batch_x)
                    else:
                        outputs = model(batch_x)

                    f_dim = -1 if args.features == "MS" else 0
                    outputs = outputs[:, -args.pred_len :, f_dim:]
                    batch_y_target = batch_y[:, -args.pred_len :, f_dim:].to(device)

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
                        "model_state_dict": model.module.state_dict() if args.use_gpu else model.state_dict(),
                        "optimizer_state_dict": model_optim.state_dict(),
                    },
                    os.path.join(temp_checkpoint_dir, "checkpoint.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(metrics=results_dict, checkpoint=checkpoint)
        else:
            train.report(metrics=results_dict, checkpoint=None)

        adjust_learning_rate(model_optim, epoch + 1, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Ray Train Script for Time Series Forecasting with DLinear on ETTh1")

    # basic config
    parser.add_argument("--train_only", action="store_true", help="perform training on full input dataset without validation")
    parser.add_argument("--smoke-test", action="store_true", default=False, help="run a quick smoke test on a small subset of data")

    # data loader args
    parser.add_argument("--root_path", type=str, default="./e2e_timeseries/dataset/", help="root path of the data file")
    parser.add_argument("--num_data_workers", type=int, default=10, help="Number of workers for PyTorch DataLoader")
    parser.add_argument(
        "--features",
        type=str,
        default="S",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate'",
    )
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location for Ray Train checkpoints")

    # forecasting task args
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")

    # DLinear specific args
    parser.add_argument("--individual", action="store_true", default=False, help="DLinear: individual layers per channel")
    # Note: enc_in is set dynamically based on features

    # optimization args
    parser.add_argument("--num_replicas", type=int, default=1, help="Number of Ray Train model replicas")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience (Note: requires custom implementation or callback)")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="optimizer learning rate")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate strategy")
    parser.add_argument("--use_amp", action="store_true", default=False, help="use automatic mixed precision training")

    # GPU / Resources
    parser.add_argument("--use_gpu", action="store_true", default=False, help="use GPU for training")

    # Other args
    parser.add_argument("--fix_seed", type=int, default=2021, help="random seed")

    args = parser.parse_args()

    # Set dataset specific args
    args.data = "ETTh1"
    args.data_path = "ETTh1.csv"
    if args.features == "S":  # S: univariate predict univariate
        args.enc_in = 1
    else:  # M or MS
        args.enc_in = 7  # ETTh1 has 7 features

    # Ensure paths are absolute
    args.root_path = os.path.abspath(args.root_path)
    args.data_path = os.path.abspath(os.path.join(args.root_path, args.data_path))
    args.checkpoints = os.path.abspath(args.checkpoints)

    # --- Smoke Test Modifications ---
    if args.smoke_test:
        print("--- RUNNING SMOKE TEST ---")
        args.train_epochs = 2
        args.batch_size = 2
        args.num_data_workers = 1

    return args


if __name__ == "__main__":
    args = parse_args()

    # === Ray Train Setup ===
    ray.init()

    scaling_config = ScalingConfig(
        num_workers=args.num_replicas,
        use_gpu=args.use_gpu,
        # resources_per_worker={"GPU": 1} if args.use_gpu else {}
    )

    # Adjust run name for smoke test
    run_name_prefix = "SmokeTest_" if args.smoke_test else ""
    run_name = f"{run_name_prefix}DLinear_{args.data}_{args.features}_{args.target}_{time.strftime('%Y%m%d_%H%M%S')}"

    run_config = RunConfig(
        storage_path=args.checkpoints,
        name=run_name,
        checkpoint_config=CheckpointConfig(num_to_keep=2, checkpoint_score_attribute="val/loss", checkpoint_score_order="min"),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=vars(args),
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # === Run Training ===
    print("Starting Ray Train job...")
    result = trainer.fit()
    print("Training finished.")

    # === Post-Training ===
    if result.best_checkpoints:
        best_checkpoint = result.get_best_checkpoint(metric="val/loss" if not args.train_only else "train/loss", mode="min")
        if best_checkpoint:
            print("Best checkpoint found:")
            print(f"  Directory: {best_checkpoint.path}")
        else:
            print("Could not retrieve the best checkpoint.")
    else:
        print("No checkpoints were saved during training.")
