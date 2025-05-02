import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear
from torch import optim
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        # Ensure result_path exists in args, provide a default if necessary
        if not hasattr(self.args, "result_path"):
            print("Warning: args.result_path not set, defaulting to './results'")
            self.args.result_path = "./results"

    def _build_model(self):
        model = DLinear.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len :, 0:]
                batch_y = batch_y[:, -self.args.pred_len :, 0:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = None, None  # Initialize
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag="val")

        # Use configured result path for checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler("cuda")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # --- AMP Training ---
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_x)
                        outputs = outputs[:, -self.args.pred_len :, 0:]
                        batch_y = batch_y[:, -self.args.pred_len :, 0:].to(self.device)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                # --- Standard Training ---
                else:
                    outputs = self.model(batch_x)
                    outputs = outputs[:, -self.args.pred_len :, 0:]
                    batch_y = batch_y[:, -self.args.pred_len :, 0:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

                # --- Logging ---
                if (i + 1) % 100 == 0:
                    print(f"	iters: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"	speed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}")
            train_loss = np.average(train_loss)

            # --- Validation and Early Stopping ---
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
                early_stopping(vali_loss, self.model, path)
            else:
                # If only training, use train_loss for early stopping
                print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f}")
                early_stopping(train_loss, self.model, path)  # Monitor train loss if no validation

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # --- Load Best Model ---
        best_model_path = os.path.join(path, "checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        else:
            print(f"Warning: Checkpoint not found at {best_model_path}. Returning model from last epoch.")

        return self.model

    def _visualize_predictions(self, batch_x_vis, true_vis, pred_vis, viz_path, batch_idx, feature_index=0):
        """Helper function to visualize predictions for a single batch item."""
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)

        # Ensure numpy arrays
        batch_x_vis = batch_x_vis.detach().cpu().numpy() if isinstance(batch_x_vis, torch.Tensor) else batch_x_vis
        true_vis = true_vis.detach().cpu().numpy() if isinstance(true_vis, torch.Tensor) else true_vis
        pred_vis = pred_vis.detach().cpu().numpy() if isinstance(pred_vis, torch.Tensor) else pred_vis

        # Basic shape and dimension checks
        if batch_x_vis.ndim < 3 or true_vis.ndim < 3 or pred_vis.ndim < 3:
            print(f"Skipping visualization for batch {batch_idx}: Input arrays need at least 3 dimensions.")
            return
        if batch_x_vis.shape[0] == 0 or true_vis.shape[0] == 0 or pred_vis.shape[0] == 0:
            print(f"Skipping visualization for batch {batch_idx}: Input arrays are empty.")
            return
        if not (batch_x_vis.shape[2] > feature_index and true_vis.shape[2] > feature_index and pred_vis.shape[2] > feature_index):
            print(f"Skipping visualization for batch {batch_idx}: Feature index {feature_index} out of bounds.")
            return

        # Concatenate history and future for ground truth and prediction
        # Use the first item in the batch (index 0) for visualization
        gt = np.concatenate((batch_x_vis[0, :, feature_index], true_vis[0, :, feature_index]), axis=0)
        pd_vis_cat = np.concatenate((batch_x_vis[0, :, feature_index], pred_vis[0, :, feature_index]), axis=0)

        # Call the visualization utility
        visual(gt, pd_vis_cat, os.path.join(viz_path, f"batch_{batch_idx}.pdf"))

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model for testing")
            model_path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # Ensure loading to correct device
                print(f"Loaded model from {model_path}")
            else:
                # Raise error instead of just printing
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Cannot perform test.")

        preds = []
        trues = []
        inputx = []

        # Configure output paths using args.result_path and setting
        base_output_path = os.path.join(self.args.result_path, setting)
        test_viz_path = os.path.join(base_output_path, "test_visuals")  # Specific folder for visualizations
        test_outputs_path = os.path.join(base_output_path, "test_outputs")  # Specific folder for numpy arrays
        results_summary_file = os.path.join(self.args.result_path, f"{setting}_results.txt")  # Unique summary file per setting

        # Create directories if they don't exist
        os.makedirs(test_viz_path, exist_ok=True)
        os.makedirs(test_outputs_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)  # Keep on device for potential loss calculation if needed later

                # --- AMP Inference ---
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_x)
                # --- Standard Inference ---
                else:
                    outputs = self.model(batch_x)

                outputs = outputs[:, -self.args.pred_len :, 0:]
                batch_y_sliced = batch_y[:, -self.args.pred_len :, 0:]

                # Detach and move to CPU for numpy conversion and metrics
                pred_batch = outputs.detach().cpu().numpy()
                true_batch = batch_y_sliced.detach().cpu().numpy()
                input_batch = batch_x.detach().cpu().numpy()

                preds.append(pred_batch)
                trues.append(true_batch)
                inputx.append(input_batch)

                # --- Visualization ---
                if i % 20 == 0:  # Visualize every 20 batches
                    # Pass relevant parts of the batch to the helper function
                    if pred_batch.shape[0] > 0:  # Check if batch has items before visualizing
                        self._visualize_predictions(
                            batch_x_vis=batch_x,  # Pass tensor, helper handles conversion
                            true_vis=batch_y_sliced,  # Pass sliced tensor
                            pred_vis=outputs,  # Pass tensor
                            viz_path=test_viz_path,
                            batch_idx=i,
                            feature_index=0,  # Assuming visualization of the first feature
                        )
                    else:
                        print(f"Skipping visualization for empty batch {i}")

        # Concatenate results from all batches
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        print("Test shapes:", preds.shape, trues.shape, inputx.shape)  # Log shapes

        # --- Calculate and Save Metrics ---
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print(f"Metrics - MSE:{mse:.7f}, MAE:{mae:.7f}, RSE:{rse:.7f}, Corr:{corr:.7f}")

        # Save metrics to a setting-specific file, overwriting previous runs for this setting
        try:
            # Ensure parent directory for the results file exists
            os.makedirs(os.path.dirname(results_summary_file), exist_ok=True)
            with open(results_summary_file, "w") as f:  # Use 'w' to overwrite
                f.write(f"Setting: {setting}\n")
                f.write(f"MSE: {mse:.7f}\n")
                f.write(f"MAE: {mae:.7f}\n")
                f.write(f"RSE: {rse:.7f}\n")
                f.write(f"Corr: {corr:.7f}\n")
                # Add other metrics if desired
                f.write(f"RMSE: {rmse:.7f}\n")
                f.write(f"MAPE: {mape:.7f}\n")
                f.write(f"MSPE: {mspe:.7f}\n")
            print(f"Results saved to {results_summary_file}")
        except IOError as e:
            print(f"Error writing results to {results_summary_file}: {e}")

        # --- Save Predictions and Ground Truth ---
        pred_file = os.path.join(test_outputs_path, "pred.npy")
        true_file = os.path.join(test_outputs_path, "true.npy")
        input_file = os.path.join(test_outputs_path, "x.npy")  # Consistent naming

        np.save(pred_file, preds)
        np.save(true_file, trues)
        np.save(input_file, inputx)
        print(f"Predictions saved to {pred_file}")
        print(f"Ground truth saved to {true_file}")
        print(f"Inputs saved to {input_file}")

        return  # Test function doesn't need to return anything explicitly
