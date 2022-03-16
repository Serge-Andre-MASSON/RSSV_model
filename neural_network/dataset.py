import numpy as np
from torch import tensor
from torch.utils.data.dataset import TensorDataset


def split(array: np.ndarray, here: int):
    train_data = array[here:]
    test_data = array[:here]
    return train_data, test_data


def dataset(x_train, y_train):
    split_here = len(x_train) // 5
    x_train, x_test = split(x_train, split_here)
    y_train, y_test = split(y_train, split_here)

    train_ds = TensorDataset(
        tensor(x_train),
        tensor(y_train),)
    test_ds = TensorDataset(
        tensor(x_test),
        tensor(y_test))
    return train_ds, test_ds
