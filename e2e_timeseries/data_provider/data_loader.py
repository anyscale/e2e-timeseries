import os
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(
        self, root_path, flag="train", size=None, features="S", data_path="ETTh1.csv", target="OT", scale=True, train_only=False, smoke_test=False
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.train_only = train_only
        self.smoke_test = smoke_test

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.abspath(os.path.join(self.root_path, self.data_path)))

        # Define borders based on train_only flag or smoke_test flag
        if self.smoke_test:
            print("--- Using smoke test data subset with Train/Val/Test splits ---")
            smoke_total_samples = 1000  # Total samples for smoke test
            # Split smoke data: 80% train, 10% val, 10% test
            smoke_val_samples = smoke_total_samples // 10
            smoke_test_samples = smoke_total_samples // 10
            smoke_train_samples = smoke_total_samples - smoke_val_samples - smoke_test_samples

            num_train = smoke_train_samples
            num_vali = smoke_val_samples
            num_test = smoke_test_samples

            # Calculate borders for the smoke test splits
            # Ensure seq_len doesn't cause negative indices
            border1s = [0, max(0, num_train - self.seq_len), max(0, num_train + num_vali - self.seq_len)]
            border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]

        elif self.train_only:
            num_train = len(df_raw)
            num_vali = 0
            num_test = 0
            # Train on all data, val/test are empty
            border1s = [0, 0, 0]
            border2s = [num_train, 0, 0]
        else:
            # Original ETTh1 split logic
            num_train = 12 * 30 * 24
            num_vali = 4 * 30 * 24
            num_test = 4 * 30 * 24
            border1s = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
            border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            # Scale using the training portion defined by border1s[0] and border2s[0]
            # This ensures the scaler is fit correctly even in smoke_test or train_only mode
            train_data_for_scaler = df_raw[cols_data if self.features != "S" else [self.target]][border1s[0] : border2s[0]]
            self.scaler.fit(train_data_for_scaler.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Turn into 1D numpy arrays
        seq_x = self.data_x[s_begin:s_end].squeeze(-1)
        seq_y = self.data_y[r_begin:r_end].squeeze(-1)

        return seq_x, seq_y

    def __len__(self):
        # # Handle case where border1/border2 might lead to empty data for val/test in train_only mode
        # if border2s[self.set_type] <= border1s[self.set_type]:
        #      return 0
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
