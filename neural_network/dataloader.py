from torch.utils.data.dataloader import DataLoader


def dataloader(train_ds, test_ds, batch_size=50):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=2 * batch_size)
    return train_dl, test_dl
