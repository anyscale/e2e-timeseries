import os
import tempfile

# Enable Ray Train V2
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import time
import warnings
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import ray
from ray import train
from ray.train import ScalingConfig, Checkpoint, CheckpointConfig, RunConfig
from ray.train.torch import TorchTrainer

from data_provider.data_factory import data_provider
from models import DLinear, NLinear
from utils.metrics import metric
from utils.tools import adjust_learning_rate

warnings.filterwarnings("ignore")



# Define model dictionary globally or pass it appropriately
model_dict = {
    "DLinear": DLinear,
    "NLinear": NLinear,
}

def train_loop_per_worker(config: dict):
    """Main training loop adapted for Ray Train workers."""
    args = argparse.Namespace(**config) # Convert dict back to Namespace for compatibility

    fix_seed = args.fix_seed if hasattr(args, 'fix_seed') else 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    device = train.torch.get_device() # FIXME
    # Let's keep manual logic for now and adapt later if needed
    if args.use_gpu:
        device = train.torch.get_device()
        # FIXME Need to handle multi-GPU specific logic if nn.DataParallel was essential
        # Ray handles distribution, so DataParallel might not be needed directly here.
    else:
        device = torch.device("cpu")

    # === Build Model ===
    model = model_dict[args.model].Model(args).float() # Instantiate the model
    model = train.torch.prepare_model(model) # Prepare model for distributed training
    model.to(device)

    # === Get Data ===
    # FIXME
    # Data provider needs to be compatible with distributed setup
    # Ray Data or manual sharding might be needed for large datasets.
    # For now, assume data_provider works correctly per worker,
    # but may load redundant data if not sharded.
    train_data, train_loader = data_provider(args, flag="train")
    if not args.train_only:
        vali_data, vali_loader = data_provider(args, flag="val")
        # test_data, test_loader = data_provider(args, flag="test") # Test data usually loaded separately

    # Prepare dataloader for distributed training
    train_loader = train.torch.prepare_data_loader(train_loader)
    if not args.train_only:
        vali_loader = train.torch.prepare_data_loader(vali_loader)
        # test_loader = train.torch.prepare_data_loader(test_loader)

    # === Optimizer and Criterion ===
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # === Early Stopping (Potentially per-worker, needs aggregation) ===
    # Early stopping logic might need adjustment for distributed setting.
    # Aggregating validation loss across workers before making a stopping decision is common.
    # For simplicity, let's keep it per-worker for now.
    # Path management also needs care in distributed setting. Checkpoints handled by Ray Train.
    # early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # === AMP Scaler ===
    if args.use_amp:
        # scaler = torch.cuda.amp.GradScaler() # Use torch's scaler
        scaler = torch.amp.GradScaler("cuda")


    # === Training Loop (Adapted from Exp_Main.train) ===
    for epoch in range(args.train_epochs):
        model.train()
        train_loss_epoch = []
        epoch_start_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # Forward pass
            if args.use_amp:
                with torch.amp.autocast("cuda"):
                    if "Linear" in args.model:
                        outputs = model(batch_x)
                    else: # Assuming non-linear models need marks/dec_inp
                         # This part needs careful review based on actual model implementations
                         # Assuming a generic call for non-linear for now
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # May need adjustment

                    f_dim = -1 if args.features == "MS" else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    loss = criterion(outputs, batch_y_target)
            else:
                if "Linear" in args.model:
                    outputs = model(batch_x)
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # May need adjustment

                f_dim = -1 if args.features == "MS" else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -args.pred_len:, f_dim:].to(device)
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

            # Optional: Log iteration loss (can be verbose in distributed setting)
            # if (i + 1) % 100 == 0:
            #     print(f"Worker {train.get_context().get_world_rank()}: Epoch {epoch+1}, Iter {i+1}, Loss: {loss.item():.7f}")

        # === End of Epoch ===
        epoch_train_loss = np.average(train_loss_epoch)
        epoch_duration = time.time() - epoch_start_time

        results_dict = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "epoch_duration_s": epoch_duration,
        }

        # === Validation (Adapted from Exp_Main.vali) ===
        if not args.train_only:
            model.eval()
            total_vali_loss = []
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device) # Target stays on device
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)

                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                    if args.use_amp:
                        with torch.amp.autocast("cuda"):
                             if "Linear" in args.model:
                                outputs = model(batch_x)
                             else:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # May need adjustment
                    else:
                        if "Linear" in args.model:
                            outputs = model(batch_x)
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # May need adjustment

                    f_dim = -1 if args.features == "MS" else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -args.pred_len:, f_dim:].to(device)

                    vali_loss = criterion(outputs, batch_y_target)
                    total_vali_loss.append(vali_loss.item()) # Collect loss per batch

            # Average validation loss for the epoch
            epoch_vali_loss = np.average(total_vali_loss)
            results_dict["vali_loss"] = epoch_vali_loss
            # FIXME: calculate other metrics like MAE, RMSE, etc.
            print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.7f}, Vali Loss: {epoch_vali_loss:.7f}")


        # === Reporting and Checkpointing ===
        # Report metrics and potentially save checkpoint
        # Checkpoint saving frequency can be controlled by RunConfig
        if train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model_optim.state_dict(),
                    },
                    os.path.join(temp_checkpoint_dir, "checkpoint.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(metrics=results_dict, checkpoint=checkpoint)
        else:
            train.report(metrics=results_dict, checkpoint=None)
            

        # Adjust learning rate (ensure adjust_learning_rate is compatible)
        adjust_learning_rate(model_optim, epoch + 1, args)

        # Note: Early stopping logic based on aggregated validation loss
        # would typically happen in the driver process after train.report,
        # but Ray Train doesn't directly support stopping from the driver yet.
        # A common pattern is to let it run for max_epochs and select the best checkpoint later.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ray Train Script for Time Series Forecasting")

    # basic config (Removed is_training, model_id, itr as they are handled differently)
    parser.add_argument("--train_only", action='store_true', help="perform training on full input dataset without validation")
    parser.add_argument("--model", type=str, default="DLinear", help="model name, options: [DLinear, NLinear]") # Adjusted default

    # data loader
    parser.add_argument("--data", type=str, default="ETTh1", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./e2e_timeseries/dataset/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument("--features", type=str, default="M", help="forecasting task, options:[M, S, MS]")
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    parser.add_argument("--freq", type=str, default="h", help="freq for time features encoding")
    parser.add_argument("--checkpoints", type=str, default="./ray_checkpoints/", help="location for Ray Train checkpoints")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")

    # Model specific args (keep relevant ones)
    parser.add_argument("--individual", action="store_true", default=False, help="DLinear: individual layers per channel")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size / number of channels")
    parser.add_argument("--c_out", type=int, default=7, help="output size") # Needed by models

    # FIXME Non-Linear model args (keep if needed, else remove)
    # parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    # parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    # parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    # parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    # parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    # parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    # parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    # parser.add_argument('--factor', type=int, default=1, help='attn factor')
    # parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
    # parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    # parser.add_argument('--activation', type=str, default='gelu', help='activation')
    # parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # optimization
    parser.add_argument("--num_workers", type=int, default=1, help="Number of Ray workers (== number of GPUs for multi-GPU)")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="optimizer learning rate")
    parser.add_argument("--loss", type=str, default="mse", help="loss function (Note: fixed to MSE in loop)")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate strategy")
    parser.add_argument("--use_amp", action="store_true", default=False, help="use automatic mixed precision training")

    # GPU / Resources
    parser.add_argument("--use_gpu", action='store_true', default=False, help="use GPU for training")
    # Removed --gpu, --use_multi_gpu, --devices as Ray handles resource allocation

    # Other args from original script (keep if needed by data_provider or models)
    parser.add_argument("--fix_seed", type=int, default=2021, help="random seed")
    # parser.add_argument("--des", type=str, default="test", help="exp description (used for logging path)")
    # parser.add_argument("--do_predict", action="store_true", help="whether to predict unseen future data")
    # parser.add_argument("--test_flop", action="store_true", default=False, help="See utils/tools for usage")


    args = parser.parse_args()

    # Ensure absolute paths for each of the paths above
    args.root_path = os.path.abspath(args.root_path)
    args.data_path = os.path.abspath(os.path.join(args.root_path, args.data_path))
    args.checkpoints = os.path.abspath(args.checkpoints)

    # === Ray Train Setup ===
    # ray.init(address="auto") # Or specify address if connecting to existing cluster
    ray.init()

    # Configure Scaling
    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        # Specify resources_per_worker if needed (e.g., {"GPU": 1})
        # Ray usually detects GPUs automatically when use_gpu=True
    )

    # Configure Run Behavior (Checkpointing, Logging)
    # Example: Keep best 2 checkpoints based on validation loss
    run_config = RunConfig(
        storage_path=args.checkpoints, # Directory to store checkpoints and logs
        name=f"{args.model}_{args.data}", # Experiment name
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            # checkpoint_score_attribute="vali_loss", # Metric to determine best checkpoint
            # checkpoint_score_order="min"           # Keep checkpoints with minimum validation loss
             checkpoint_score_attribute="train_loss", # Metric to determine best checkpoint (use train if no vali)
             checkpoint_score_order="min"           # Keep checkpoints with minimum train loss
        ),
    )
    if not args.train_only:
         run_config.checkpoint_config.checkpoint_score_attribute="vali_loss"
         run_config.checkpoint_config.checkpoint_score_order="min"


    # === Initialize TorchTrainer ===
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=vars(args), # Pass parsed args to workers
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # === Run Training ===
    print("Starting Ray Train job...")
    result = trainer.fit()
    print("Training finished.")

    # === Post-Training (Optional: Load best checkpoint and test/predict) ===
    best_checkpoint = result.best_checkpoints # Get path and metric of the best checkpoint
    print(f"Best checkpoint information: {best_checkpoint}")

    # Example: Load best checkpoint and run test/predict
    # This would require adapting the test/predict logic from Exp_Main
    # similar to how the train loop was adapted.
    # best_checkpoint_path = best_checkpoint[0][0].path # Access path correctly
    # print(f"Path to best checkpoint: {best_checkpoint_path}")
    # Add logic here to load the checkpoint and run evaluation if needed.
