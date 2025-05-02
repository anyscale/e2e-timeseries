import argparse
import random

import numpy as np
import torch
from exp.exp_main import Exp_Main

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="DLinear for Time Series Forecasting")

    # basic config
    parser.add_argument("--is_training", type=int, default=True, help="status")
    parser.add_argument(
        "--train_only", type=bool, required=False, default=False, help="perform training on full input dataset without validation and testing"
    )
    parser.add_argument("--smoke_test", action="store_true", default=False, help="Run on a small subset of data for quick testing")
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument("--model", type=str, default="DLinear", help="model name")

    # data loader
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
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")
    parser.add_argument("--result_path", type=str, default="./results/", help="location for output results")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length (kept for data loading compatibility)")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")

    # DLinear
    parser.add_argument("--individual", action="store_true", default=False, help="DLinear: a linear layer for each variate(channel) individually")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size / number of channels")

    # optimization
    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times (reduced default)")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    args.model = "DLinear"

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = "{}_{}_{}_ft{}_sl{}_pl{}_ind{}_{}_{}".format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.individual,
                args.des,
                ii,
            )

            exp = Exp(args)
            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)

            if not args.train_only:
                print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
                exp.test(setting)

            torch.cuda.empty_cache()
