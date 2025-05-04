import ray
from data_provider.data_loader import TimeSeriesDataset


def data_provider(args, flag):
    data_set = TimeSeriesDataset(
        root_path=args.root_path, flag=flag, size=[args.seq_len, args.label_len, args.pred_len], smoke_test=getattr(args, "smoke_test", False)
    )
    print(f"{flag} data set length: {len(data_set)}")
    return ray.data.from_torch(data_set)
