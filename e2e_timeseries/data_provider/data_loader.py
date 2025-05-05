import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class TimeSeriesDataset(Dataset):
    def __init__(self, root_path, flag="train", size=None, smoke_test=False):
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
        self.target = "OT"  # col name of the target variable (oil temperature)
        self.scale = True
        self.data_path = "ETTh1.csv"  # fixme read from argsS

        self.root_path = root_path
        self.smoke_test = smoke_test

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), header=0)
        # Define time periods in hours
        hours_per_day = 24
        days_per_month = 30

        # Define dataset splits in months
        train_months = 12
        val_months = 4
        test_months = 4

        # Calculate boundaries for each split in hours
        train_hours = train_months * days_per_month * hours_per_day
        val_hours = val_months * days_per_month * hours_per_day
        test_hours = test_months * days_per_month * hours_per_day

        # Define borders for train, validation and test sets
        border1s = [
            0,  # train start
            train_hours - self.seq_len,  # val start
            train_hours + val_hours - self.seq_len,  # test start
        ]

        border2s = [
            train_hours,  # train end
            train_hours + val_hours,  # val end
            train_hours + val_hours + test_hours,  # test end
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw[[self.target]]

        train_data = df_data[border1s[0] : border2s[0]]
        self.scaler.fit(train_data.values)
        # fixme: we shouldn't transform the whole dataset, just the oil temperature data?
        data = self.scaler.transform(df_data.values)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

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

    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # Squeeze the arrays to make them 1D
        seq_x = seq_x.squeeze()
        seq_y = seq_y.squeeze()

        # return {'x': seq_x, 'y': seq_y}
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
