from data_provider.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader

def data_provider(args, flag):
    # Determine Data class based on flag
    if flag == 'test':
        Data = Dataset_ETT_hour
        batch_size = args.batch_size
        shuffle_flag = False
        drop_last = False
    else: # flag == 'train' or 'val'
        Data = Dataset_ETT_hour
        batch_size = args.batch_size
        shuffle_flag = True
        drop_last = True

    timeenc = 0 if args.embed != "timeF" else 1
    freq = args.freq
    train_only = args.train_only

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only,
    )
    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_data_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
