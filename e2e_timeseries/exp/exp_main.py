import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear
from torch import optim
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, test_params_flop, visual

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

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

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag="val")
            test_data, test_loader = self._get_data(flag="test")

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

                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss
                    )
                )
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print(f"Warning: Checkpoint not found at {best_model_path}. Returning model without loading best state.")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            model_path = os.path.join("./checkpoints/" + setting, "checkpoint.pth")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
            else:
                print(f"Error: Model checkpoint not found at {model_path}. Cannot perform test.")
                return

        preds = []
        trues = []
        inputx = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if (
                        input.shape[0] > 0
                        and true.shape[0] > 0
                        and pred.shape[0] > 0
                        and input.shape[2] > 0
                        and true.shape[2] >= 0
                        and pred.shape[2] >= 0
                    ):
                        feature_index_to_visualize = -1 if true.ndim > 2 else 0
                        gt = np.concatenate((input[0, :, feature_index_to_visualize], true[0, :, feature_index_to_visualize]), axis=0)
                        pd_vis = np.concatenate((input[0, :, feature_index_to_visualize], pred[0, :, feature_index_to_visualize]), axis=0)
                        visual(gt, pd_vis, os.path.join(folder_path, str(i) + ".pdf"))
                    else:
                        print(f"Skipping visualization for batch {i} due to incompatible shapes.")

        if self.args.test_flop:
            if hasattr(self.model, "seq_len") and hasattr(self.model, "pred_len"):
                test_params_flop((self.args.seq_len, self.args.enc_in))
            else:
                print("Skipping FLOP test: Model attributes not found.")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        try:
            f = open("result.txt", "a")
            f.write(setting + "  \n")
            f.write("mse:{}, mae:{}, rse:{}, corr:{}".format(mse, mae, rse, corr))
            f.write("\n")
            f.write("\n")
            f.close()
        except IOError as e:
            print(f"Error writing to result.txt: {e}")

        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        np.save(folder_path + "x.npy", inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))
            else:
                print(f"Error: Model checkpoint not found at {best_model_path}. Cannot perform prediction.")
                return

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        if len(preds) > 0:
            preds = np.concatenate(preds, axis=0)
        else:
            print("Warning: No predictions were generated.")
            return

        if pred_data and hasattr(pred_data, "scale") and pred_data.scale and hasattr(pred_data, "inverse_transform"):
            preds = pred_data.inverse_transform(preds)
        elif pred_data and hasattr(pred_data, "scale") and pred_data.scale:
            print("Warning: Data is scaled but inverse_transform method not found or failed. Saving scaled predictions.")

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        if pred_data and hasattr(pred_data, "future_dates") and hasattr(pred_data, "cols"):
            try:
                if preds.ndim >= 3 and preds.shape[0] > 0:
                    first_pred_sample = preds[0]
                    if len(pred_data.future_dates) == first_pred_sample.shape[0]:
                        dates_col = np.transpose([pred_data.future_dates])
                        df_data = np.append(dates_col, first_pred_sample, axis=1)
                        if len(pred_data.cols) == first_pred_sample.shape[1]:
                            df_cols = [pred_data.timeenc_col] + pred_data.cols if hasattr(pred_data, "timeenc_col") else ["date"] + pred_data.cols
                            if len(df_cols) == df_data.shape[1]:
                                pd.DataFrame(df_data, columns=df_cols).to_csv(folder_path + "real_prediction.csv", index=False)
                            else:
                                print(
                                    f"Warning: Column mismatch for CSV saving. Expected {df_data.shape[1]} columns, got {len(df_cols)}. Saving NPY only."
                                )
                        else:
                            print(
                                f"Warning: Feature column count mismatch for CSV saving. Expected {first_pred_sample.shape[1]} features, got {len(pred_data.cols)}. Saving NPY only."
                            )
                    else:
                        print(
                            f"Warning: Mismatch between future_dates length ({len(pred_data.future_dates)}) and prediction length ({first_pred_sample.shape[0]}). Cannot save CSV."
                        )
                else:
                    print("Warning: Prediction array has unexpected dimensions or is empty. Cannot save CSV.")
            except Exception as e:
                print(f"Error saving prediction CSV: {e}. Saving NPY file only.")
        else:
            print("Warning: Missing attributes (future_dates or cols) in pred_data. Cannot save CSV.")

        return
