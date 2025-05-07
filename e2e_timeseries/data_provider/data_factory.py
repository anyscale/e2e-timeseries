from data_provider.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader


def data_provider(config: dict, flag: str):
    # Determine Data class based on flag
    if flag in ["test", "val"]:
        shuffle_flag = False
        drop_last = False
    else:  # flag == 'train' or 'val'
        shuffle_flag = True

    train_only = config["train_only"]
    smoke_test = config["smoke_test"] if "smoke_test" in config else False

    # Override drop_last for smoke test to prevent data loss
    if smoke_test:
        drop_last = False

    data_set = Dataset_ETT_hour(
        root_path=config["root_path"],
        data_path=config["data_path"],
        flag=flag,
        size=[config["seq_len"], config["label_len"], config["pred_len"]],
        features=config["features"],
        target=config["target"],
        train_only=train_only,
        smoke_test=smoke_test,
    )
    print(f"{flag} subset size: {len(data_set)}")

    data_loader = DataLoader(
        data_set, batch_size=config["batch_size"], shuffle=shuffle_flag, num_workers=config["num_data_workers"], drop_last=drop_last
    )
    assert len(data_loader) > 0
    return data_loader, data_set
