import random
import math


def train_test_split(data,train_test_percentage : float):
    random.shuffle(data)
    data_len = len(data)
    train_len = math.floor(data_len*train_test_percentage)
    train_data = data[:train_len]
    test_data = data[train_len:]
    return train_data, test_data

