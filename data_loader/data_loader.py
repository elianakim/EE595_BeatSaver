from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import math
import copy

from .BeatChange_Dataset import BeatChange_Dataset
from .BeatType_Dataset import BeatType_Dataset

import conf

def split_data(entire_data, valid_split, test_split, train_max_rows, valid_max_rows, test_max_rows):
    valid_size = math.floor(len(entire_data) * valid_split)
    test_size = math.floor(len(entire_data) * test_split)

    train_size = len(entire_data) - valid_size - test_size

    assert (train_size >= 0 and valid_size >= 0 and test_size >= 0)

    train_data, valid_data, test_data = torch.utils.data.random_split(entire_data,
                                                                      [train_size, valid_size, test_size])

    if len(entire_data) > train_max_rows:
        train_data = torch.utils.data.Subset(train_data, range(train_max_rows))
    if len(valid_data) > valid_max_rows:
        valid_data = torch.utils.data.Subset(valid_data, range(valid_max_rows))
    if len(test_data) > test_max_rows:
        test_data = torch.utils.data.Subset(test_data, range(test_max_rows))

    return train_data, valid_data, test_data

def single_domain_data_loader(file_path, batch_size, train_max_rows=np.inf, valid_max_rows=np.inf, test_max_rows=np.inf,
                              valid_split=0.1, test_split=0.1):
    entire_datasets = []
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    st = time.time()

    if conf.args.dataset in ['beat_original']:
        if conf.args.type in ['beat_change']:
            train_data = BeatChange_Dataset(file=file_path)
        elif conf.args.type in ['beat_type']:
            train_data = BeatType_Dataset(file=file_path)

    # split dataset into train, valid, and test
    total_len = len(train_data)
    train_data, valid_data, test_data = split_data(train_data, valid_split, test_split,
                                                   train_max_rows, valid_max_rows, test_max_rows)

    print('Data_loader len: {:d}\t# Train: {:d}\t# Valid: {:d}\t%# Test: {:d}'.format(
        total_len, len(train_data), len(valid_data), len(test_data)))

    # dataset to dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                             num_workers=8, drop_last=True, pin_memory=False)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True,
                             num_workers=8, drop_last=True, pin_memory=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                             num_workers=8, drop_last=True, pin_memory=False)

    data_loader = {
        'train' : train_data_loader,
        'valid' : valid_data_loader,
        'test' : test_data_loader,
    }
    return data_loader

def data_loader_for_demo(file_path, batch_size = 1):
    if conf.args.dataset in ['beat_original']:
        if conf.args.type in ['beat_change']:
            train_data = BeatChange_Dataset(file=file_path)
        elif conf.args.type in ['beat_type']:
            train_data = BeatType_Dataset(file=file_path)

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                             num_workers=8, drop_last=True, pin_memory=False)
    return data_loader




if __name__ == '__main__':
    pass