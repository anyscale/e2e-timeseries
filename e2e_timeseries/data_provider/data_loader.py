import os
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


class TimeSeriesDataset(Dataset):
    def __init__(self, root_path, flag="train", size=None, timeenc=0, freq="h", smoke_test=False):
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

        self.features = "S"
        self.target = "OT"
        self.scale = True
        self.data_path = "ETTh1.csv"

        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        # Store smoke_test flag
        self.smoke_test = smoke_test

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[[self.target]]

        train_data = df_data[border1s[0] : border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # Apply smoke test subsetting if enabled
        if self.smoke_test:
            print(f"Smoke test active for flag '{self.set_type}', reducing dataset size.")
            # Calculate desired length for ~500 samples
            target_len = 500 + self.seq_len + self.pred_len - 1
            # Ensure we don't exceed original length
            actual_len = min(target_len, len(self.data_x))

            if actual_len < self.seq_len + self.pred_len:
                warnings.warn(
                    "Smoke test subsetting resulted in insufficient data for the required sequence and prediction lengths. Disabling subsetting for this split."
                )
            else:
                print(f"Original length: {len(self.data_x)}, Reduced length: {actual_len}")
                self.data_x = self.data_x[:actual_len]
                self.data_y = self.data_y[:actual_len]
                self.data_stamp = self.data_stamp[:actual_len]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
