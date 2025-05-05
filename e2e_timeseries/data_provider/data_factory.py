import numpy as np
import ray
from data_provider.data_loader import TimeSeriesDataset


def data_provider(args, flag):
    data_set = TimeSeriesDataset(
        root_path=args.root_path, flag=flag, size=[args.seq_len, args.label_len, args.pred_len], smoke_test=getattr(args, "smoke_test", False)
    )
    print(f"{flag} data set length: {len(data_set)}")
    ds = ray.data.from_torch(data_set)
    print(f"example ds stage 1: {ds.show(1)}")
    # split the "item" key into two keys "x" and "y"
    # ds = ds.map(lambda item: {'x': item['item']['x'], 'y': item['item']['y']})

    def process_item(item):
        x = item["item"][0]
        y = item["item"][1]
        # gotcha: Ray Data converts the numpy arrays into lists, so we need to convert them back
        x = np.array(x)
        y = np.array(y)
        print(f"x: {x}")
        assert type(x) == np.ndarray, f"x is not a numpy array: {type(x)}"
        assert type(y) == np.ndarray, f"y is not a numpy array: {type(y)}"
        return {"x": x, "y": y}

    ds = ds.map(process_item)
    print(f"example ds stage 2: {ds.show(1)}")
    raise Exception(f"example ds stage 2: {ds.show(1)}")
    return ds
