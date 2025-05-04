import argparse
import os
import time
import warnings

import numpy as np
import ray
import ray.train
import torch
import torch.nn as nn
from models import DLinear
from ray.train import Checkpoint
from ray.train.torch import prepare_model, prepare_optimizer
from torch import optim
from utils.metrics import metric
from utils.tools import visual

warnings.filterwarnings("ignore")


def visualize_predictions(batch_x_vis, true_vis, pred_vis, viz_path, batch_idx, feature_index=0, scaler=None):
    """Helper function to visualize predictions for a single batch item.
    Added scaler argument for inverse transform.
    """
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)

    # Ensure numpy arrays on CPU
    batch_x_vis = batch_x_vis.detach().cpu().numpy() if isinstance(batch_x_vis, torch.Tensor) else np.asarray(batch_x_vis)
    true_vis = true_vis.detach().cpu().numpy() if isinstance(true_vis, torch.Tensor) else np.asarray(true_vis)
    pred_vis = pred_vis.detach().cpu().numpy() if isinstance(pred_vis, torch.Tensor) else np.asarray(pred_vis)

    # Inverse transform if scaler is provided
    if scaler is not None:
        try:
            num_features = scaler.n_features_in_
            # Reshape assuming [batch, time, features]
            original_shape_x = batch_x_vis.shape
            original_shape_true = true_vis.shape
            original_shape_pred = pred_vis.shape

            # Reshape for scaler (needs 2D: [samples, features])
            batch_x_vis_flat = batch_x_vis.reshape(-1, num_features)
            true_vis_flat = true_vis.reshape(-1, num_features)
            pred_vis_flat = pred_vis.reshape(-1, num_features)

            # Apply inverse transform
            batch_x_vis_inv = scaler.inverse_transform(batch_x_vis_flat).reshape(original_shape_x)
            true_vis_inv = scaler.inverse_transform(true_vis_flat).reshape(original_shape_true)
            pred_vis_inv = scaler.inverse_transform(pred_vis_flat).reshape(original_shape_pred)

            # Use inverse transformed data for plotting
            batch_x_vis = batch_x_vis_inv
            true_vis = true_vis_inv
            pred_vis = pred_vis_inv
            plot_title_suffix = " (Inversed)"
        except Exception as e:
            print(f"Warning: Scaler inverse_transform failed during visualization: {e}. Plotting scaled data.")
            plot_title_suffix = " (Scaled)"
    else:
        plot_title_suffix = " (Scaled)"

    # Basic shape and dimension checks
    # Assuming input shapes like [batch, time, features]
    if batch_x_vis.ndim < 3 or true_vis.ndim < 2 or pred_vis.ndim < 2:  # Adjusted true/pred dims
        print(f"Skipping visualization for batch {batch_idx}: Input array dimension mismatch.")
        return
    if batch_x_vis.shape[0] == 0 or true_vis.shape[0] == 0 or pred_vis.shape[0] == 0:
        print(f"Skipping visualization for batch {batch_idx}: Input arrays are empty.")
        return

    num_features_x = batch_x_vis.shape[-1]
    num_features_true = true_vis.shape[-1]
    num_features_pred = pred_vis.shape[-1]

    if not (num_features_x > feature_index and num_features_true > feature_index and num_features_pred > feature_index):
        print(f"Skipping visualization for batch {batch_idx}: Feature index {feature_index} out of bounds.")
        return

    # Concatenate history and future for ground truth and prediction
    # Use the first item in the batch (index 0) for visualization
    history = batch_x_vis[0, :, feature_index]
    true_future = true_vis[0, :, feature_index]
    pred_future = pred_vis[0, :, feature_index]

    gt = np.concatenate((history, true_future), axis=0)
    pd_vis_cat = np.concatenate((history, pred_future), axis=0)

    # Call the visualization utility
    plot_path = os.path.join(viz_path, f"batch_{batch_idx}_feat_{feature_index}{plot_title_suffix}.pdf")
    visual(gt, pd_vis_cat, plot_path)


# --- Ray Train Worker Function ---
def train_loop_per_worker(config: dict):
    """Ray Train training loop function.

    Args:
        config: Configuration dictionary passed by TorchTrainer.
                Expected keys: lr, epochs, batch_size, seq_len, pred_len,
                               enc_in, individual, loss, lradj,
                               use_amp, patience, model_id, data,
                               features, root_path, data_path, target,
                               train_only (bool)
    """
    # --- Setup ---
    args = argparse.Namespace(**config)

    # --- Model, Optimizer, Criterion ---
    # Build model within the worker
    model = DLinear.Model(args).float()
    # Prepare model for distributed training
    model = prepare_model(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = prepare_optimizer(optimizer)

    # --- Loss ---
    if args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss function '{args.loss}' not implemented.")

    # --- Data ---
    # Get Ray Dataset shards for this worker
    train_ds_shard = ray.train.get_dataset_shard("train")
    vali_ds_shard = ray.train.get_dataset_shard("val") if not args.train_only else None

    # --- Early Stopping ---
    best_vali_loss = float("inf")
    patience_counter = 0

    # --- AMP ---
    if args.use_amp:
        scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        train_loss_epoch = []
        epoch_time = time.time()
        num_batches = 0
        # Ray Data Integration: Iterate over batches
        # Use `iter_torch_batches` for automatic batching and device placement
        for batch in train_ds_shard.iter_torch_batches(batch_size=args.batch_size, dtypes=torch.float, device=ray.train.torch.get_device()):
            num_batches += 1
            optimizer.zero_grad()
            batch_x = batch["x"]
            batch_y = batch["y"]

            # --- AMP Training ---
            if args.use_amp:
                with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(batch_x)
                    outputs = outputs[:, -args.pred_len :, 0]
                    batch_y = batch_y[:, -args.pred_len :, 0]
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # --- Standard Training ---
            else:
                outputs = model(batch_x)
                outputs = outputs[:, -args.pred_len :, 0]
                batch_y = batch_y[:, -args.pred_len :, 0]
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            train_loss_epoch.append(loss.item())

        avg_train_loss = np.average(train_loss_epoch)
        epoch_duration = time.time() - epoch_time
        print(f"Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.7f} | Duration: {epoch_duration:.4f}s")

        # --- Validation ---
        avg_vali_loss = None
        if vali_ds_shard:
            model.eval()
            vali_loss_epoch = []
            with torch.no_grad():
                for batch in vali_ds_shard.iter_torch_batches(batch_size=args.batch_size, dtypes=torch.float, device=ray.train.torch.get_device()):
                    batch_x = batch["x"]
                    batch_y = batch["y"]

                    if args.use_amp:
                        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                            outputs = model(batch_x)
                    else:
                        outputs = model(batch_x)

                    outputs = outputs[:, -args.pred_len :, :]  # Slice
                    batch_y = batch_y[:, -args.pred_len :, :]  # Slice

                    loss = criterion(outputs, batch_y)
                    vali_loss_epoch.append(loss.item())

            avg_vali_loss = np.average(vali_loss_epoch)
            print(f"Epoch: {epoch + 1} | Vali Loss: {avg_vali_loss:.7f}")
            model.train()  # Set back to train mode

        # --- Reporting & Checkpointing ---
        metrics = {"train/loss": avg_train_loss}
        checkpoint_metric_name = "train/loss"
        checkpoint_metric_value = avg_train_loss

        if avg_vali_loss is not None:
            metrics["val/loss"] = avg_vali_loss
            checkpoint_metric_name = "val/loss"
            checkpoint_metric_value = avg_vali_loss

        # Simple Early Stopping Logic
        should_stop = False
        if checkpoint_metric_value < best_vali_loss:
            best_vali_loss = checkpoint_metric_value
            patience_counter = 0
            # Save checkpoint only if it's the best so far based on validation/train loss
            checkpoint = Checkpoint.from_dict(
                dict(
                    epoch=epoch,
                    model_state_dict=model.module.state_dict(),  # Use .module with prepare_model
                    optimizer_state_dict=optimizer.state_dict(),
                )
            )
            ray.train.report(metrics, checkpoint=checkpoint)
            print(f"Epoch {epoch + 1}: Checkpoint saved. {checkpoint_metric_name}: {checkpoint_metric_value:.7f}")
        else:
            patience_counter += 1
            ray.train.report(metrics)  # Report metrics even without checkpoint
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement.")
                should_stop = True

        # Adjust LR (TODO: Implement adjust_learning_rate logic if needed)
        # adjust_learning_rate(optimizer, epoch + 1, args)

        if should_stop:
            break

    print(f"Training loop finished after {epoch + 1} epochs for {args.model_id}.")


def acquire_device(use_gpu_flag):  # TODO: might not be necessary with Ray Train
    if use_gpu_flag and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for testing")
    else:
        if use_gpu_flag:
            print("Warning: --use_gpu flag was set, but CUDA is not available. Using CPU for testing.")
        device = torch.device("cpu")
        print("Using CPU for testing")
    return device


def run_testing(args, test_checkpoint_path, test_ds, scaler=None):
    if test_ds is None:
        print("Error: Ray Dataset for testing not provided.")
        return
    if not test_checkpoint_path or not os.path.exists(test_checkpoint_path):
        print(f"Error: Test checkpoint path not found or invalid: {test_checkpoint_path}")
        return
    if scaler is None:
        print("Warning: Scaler not provided for test function. Results will be in scaled format.")

    device = acquire_device(args.use_gpu)

    print(f"Loading model from checkpoint: {test_checkpoint_path}")
    checkpoint = Checkpoint.from_directory(test_checkpoint_path)
    model = DLinear.Model(args).float()
    model_state_dict = checkpoint.to_dict().get("model_state_dict")

    if model_state_dict:
        adjusted_state_dict = {}
        for k, v in model_state_dict.items():
            name = k.replace("_orig_mod.", "")
            adjusted_state_dict[name] = v
        model.load_state_dict(adjusted_state_dict)
        print("Successfully loaded model state dict from checkpoint.")
    else:
        print(f"Warning: Checkpoint at {test_checkpoint_path} did not contain 'model_state_dict'. Cannot perform testing.")
        return

    model.to(device)
    model.eval()

    preds = []
    trues = []
    inputx = []

    setting = "{}_{}_{}_ft{}_sl{}_pl{}_ind{}_{}_raytest".format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.individual,
        args.exp_name,
    )

    base_output_path = os.path.join(args.result_path, setting)
    test_viz_path = os.path.join(base_output_path, "test_visuals_ray")
    test_outputs_path = os.path.join(base_output_path, "test_outputs_ray")
    results_summary_file = os.path.join(args.result_path, f"{setting}_results_ray.txt")

    os.makedirs(test_viz_path, exist_ok=True)
    os.makedirs(test_outputs_path, exist_ok=True)

    with torch.no_grad():
        batch_idx = 0
        for batch in test_ds.iter_torch_batches(batch_size=args.batch_size, dtypes=torch.float, device=device):
            batch_x = batch["x"]
            batch_y = batch["y"]

            if args.use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(batch_x)
            else:
                outputs = model(batch_x)

            outputs = outputs[:, -args.pred_len :, :]
            batch_y_sliced = batch_y[:, -args.pred_len :, :]

            pred_batch = outputs.detach().cpu().numpy()
            true_batch = batch_y_sliced.detach().cpu().numpy()
            input_batch = batch_x.detach().cpu().numpy()

            preds.append(pred_batch)
            trues.append(true_batch)
            inputx.append(input_batch)

            if batch_idx % 20 == 0:
                if pred_batch.shape[0] > 0:
                    visualize_predictions(
                        batch_x_vis=batch_x,
                        true_vis=batch_y_sliced,
                        pred_vis=outputs,
                        viz_path=test_viz_path,
                        batch_idx=batch_idx,
                        feature_index=0,
                        scaler=scaler,
                    )
                else:
                    print(f"Skipping visualization for empty batch {batch_idx}")
            batch_idx += 1

    if not preds:
        print("No predictions generated during testing.")
        return

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0)
    print("Test shapes:", preds.shape, trues.shape, inputx.shape)

    if scaler is not None:
        try:
            num_features = scaler.n_features_in_
            preds_inv = scaler.inverse_transform(preds.reshape(-1, num_features)).reshape(preds.shape)
            trues_inv = scaler.inverse_transform(trues.reshape(-1, num_features)).reshape(trues.shape)
            inputx_inv = scaler.inverse_transform(inputx.reshape(-1, num_features)).reshape(inputx.shape)
            print("Inverse transform applied to outputs.")
        except Exception as e:
            print(f"Warning: Scaler inverse_transform failed for final outputs: {e}. Metrics and saved files will use scaled data.")
            preds_inv, trues_inv, inputx_inv = preds, trues, inputx
    else:
        preds_inv, trues_inv, inputx_inv = preds, trues, inputx

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds_inv, trues_inv)
    print(f"Metrics (Ray) - MSE:{mse:.7f}, MAE:{mae:.7f}, RSE:{rse:.7f}, Corr:{corr:.7f}")

    try:
        os.makedirs(os.path.dirname(results_summary_file), exist_ok=True)
        with open(results_summary_file, "w") as f:
            f.write(f"Setting: {setting} (Ray)\n")
            f.write(f"Test Checkpoint: {test_checkpoint_path}\n")
            f.write(f"MSE: {mse:.7f}\n")
            f.write(f"MAE: {mae:.7f}\n")
            f.write(f"RSE: {rse:.7f}\n")
            f.write(f"Corr: {corr:.7f}\n")
            f.write(f"RMSE: {rmse:.7f}\n")
            f.write(f"MAPE: {mape:.7f}\n")
            f.write(f"MSPE: {mspe:.7f}\n")
        print(f"Results saved to {results_summary_file}")
    except IOError as e:
        print(f"Error writing results to {results_summary_file}: {e}")

    pred_file = os.path.join(test_outputs_path, "pred.npy")
    true_file = os.path.join(test_outputs_path, "true.npy")
    input_file = os.path.join(test_outputs_path, "x.npy")

    np.save(pred_file, preds_inv)
    np.save(true_file, trues_inv)
    np.save(input_file, inputx_inv)
    print(f"Predictions (inverse-transformed) saved to {pred_file}")
    print(f"Ground truth (inverse-transformed) saved to {true_file}")
    print(f"Inputs (inverse-transformed) saved to {input_file}")

    print("Testing complete.")
