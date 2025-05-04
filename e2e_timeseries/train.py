import argparse
import os
import random

import numpy as np
import ray
import torch
from data_provider.data_factory import data_provider
from exp.exp_main import run_testing, train_loop_per_worker
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="DLinear Time Series Forecasting with Ray Train")

    parser.add_argument("--is_training", type=int, default=1, help="status (set to 1 for training, 0 for testing only)")
    parser.add_argument(
        "--train_only", type=bool, required=False, default=False, help="perform training on full input dataset without validation and testing"
    )
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run on a small subset of data for quick testing")
    parser.add_argument("--model_id", type=str, default="DLinear_ETTh1", help="model id")
    parser.add_argument("--model", type=str, default="DLinear", help="model name")

    parser.add_argument("--data", type=str, default="ETTh1", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./data/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    parser.add_argument("--result_path", type=str, default="./ray_results/", help="location for output results (metrics, test outputs)")
    parser.add_argument("--exp_name", type=str, default="dlinear_train", help="Ray experiment name (used for checkpoint/result dir)")

    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--label_len", type=int, default=48, help="start token length (kept for data loading compatibility, but not directly used in Ray DLinear)"
    )
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")

    parser.add_argument("--individual", action="store_true", default=False, help="DLinear: a linear layer for each variate(channel) individually")

    parser.add_argument("--num_workers", type=int, default=1, help="Ray Train num workers (data loading happens in train loop)")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

    parser.add_argument("--use_gpu", action="store_true", default=False, help="use GPU for workers")

    args = parser.parse_args()

    if args.features == "S" or args.features == "MS":
        args.enc_in = 1
    elif args.features == "M":
        print("Warning: Assuming 7 features for multivariate ('M') setting based on ETTh1. Adjust if using different data.")
        args.enc_in = 7
    else:
        raise ValueError(f"Unsupported feature type: {args.features}")

    if args.smoke_test:
        print("Smoke test enabled: Reducing epochs and Ray workers.")
        args.train_epochs = 1
        args.num_workers = 1

    args.root_path = os.path.abspath(args.root_path)
    args.result_path = os.path.abspath(args.result_path)

    os.makedirs(args.result_path, exist_ok=True)

    return args


if __name__ == "__main__":
    args = parse_args()

    ray.init(ignore_reinit_error=True)

    print("Args for Ray Train experiment:")
    print(args)

    train_ds, train_scaler = data_provider(args, flag="train", return_scaler=True)
    val_ds = data_provider(args, flag="val")
    test_ds = data_provider(args, flag="test")

    datasets = {"train": train_ds}
    if val_ds and not args.train_only:
        datasets["val"] = val_ds

    print("Dataset creation complete.")

    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
    )

    train_loop_config = vars(args)

    run_config = RunConfig(
        name=args.exp_name,
        storage_path=args.result_path,
        checkpoint_config=CheckpointConfig(
            num_to_keep=1, checkpoint_score_attribute="val_loss" if "val" in datasets else "train_loss", checkpoint_score_order="min"
        ),
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets=datasets,
        run_config=run_config,
    )

    best_checkpoint_path = None
    if args.is_training:
        print("\n>>> Starting Ray Train training...")
        result: ray.train.Result = trainer.fit()
        print("Training finished.")

        print(f"\nTraining Result: {result}")
        if result.metrics:
            print(f"Final reported metrics: {result.metrics}")
        if result.checkpoint:
            print(f"Best checkpoint saved at: {result.checkpoint.path}")
            best_checkpoint_path = result.checkpoint.path
        else:
            print("No checkpoint reported by the trainer.")
    else:
        print("Skipping training as is_training=0.")
        if args.result_path and args.exp_name:
            potential_checkpoint_dir = os.path.join(args.result_path, args.exp_name)
            if os.path.isdir(potential_checkpoint_dir):
                try:
                    latest_checkpoint = Checkpoint.from_directory(potential_checkpoint_dir)
                    if latest_checkpoint:
                        best_checkpoint_path = latest_checkpoint.path
                        print(f"Found potential checkpoint for testing: {best_checkpoint_path}")
                    else:
                        print(f"Could not find valid checkpoint in {potential_checkpoint_dir}")
                except Exception as e:
                    print(f"Error trying to load checkpoint from {potential_checkpoint_dir}: {e}")
            else:
                print(f"Experiment directory {potential_checkpoint_dir} not found for potential checkpoint.")
        if not best_checkpoint_path:
            print("Warning: is_training=0 and no valid checkpoint path could be determined. Cannot proceed to testing.")

    if not args.train_only:
        if best_checkpoint_path:
            if test_ds:
                print("\n>>> Starting testing using the best checkpoint...")
                run_testing(args=args, test_checkpoint_path=best_checkpoint_path, test_ds=test_ds, scaler=train_scaler)
            else:
                print("Skipping testing: Test dataset not available.")
        else:
            print("Skipping testing: No valid checkpoint path found or training did not produce one.")
    else:
        print("Skipping testing as train_only is True.")

    ray.shutdown()
    print("\nRay Train script finished.")
