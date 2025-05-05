from data_provider.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader


def data_provider(args, flag):
    # Determine Data class based on flag
    if flag == "test":
        Data = Dataset_ETT_hour
        batch_size = args.batch_size
        shuffle_flag = False
        drop_last = False
    else:  # flag == 'train' or 'val'
        Data = Dataset_ETT_hour
        batch_size = args.batch_size
        shuffle_flag = True
        drop_last = True

    train_only = args.train_only
    smoke_test = args.smoke_test if hasattr(args, "smoke_test") else False

    # Override drop_last for smoke test to prevent data loss
    if smoke_test:
        drop_last = False

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        train_only=train_only,
        smoke_test=smoke_test,
    )
    print(flag, len(data_set))

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_data_workers, drop_last=drop_last)
    assert len(data_loader) > 0
    return data_loader, data_set
