from data_provider.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader


def data_provider(args, flag):
    smoke_test = getattr(args, "smoke_test", False)

    if flag == "test":
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = TimeSeriesDataset(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        smoke_test=smoke_test,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)
    return data_set, data_loader
