import os

import torch
import pandas as pd
import time
import numpy as np
import sys

sys.path.append('..')
import conf

opt = conf.BeatChange_Opt
WIN_LEN = opt['seq_len']
OVERLAPPING = opt['overlap_ratio'] # overlapping window

class BeatChange_Dataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/20211211_debug_hr/accgyro.csv'):
        print('Loading data...')

        st = time.time()

        self.features = None
        self.class_labels = None
        self.dataset = None
        self.classes = conf.BeatChange_Opt['classes']

        self.df = pd.read_csv(file)
        self.preprocessing()
        ppt = time.time()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):
        self.features = []
        self.class_labels = []
        pt = 0
        while pt + WIN_LEN <= len(self.df):
            bt = time.time()
            # decide label
            labels = self.df.iloc[pt : pt+WIN_LEN, opt['label_index']].values
            # TODO: ADD DOMAIN. If domain has changed, skip the data
            # if beat change included in the window, label as 'Change'
            # else, 'Not Change'
            label = 'Not Change'
            for l in labels:
                if l != 'None':
                    label = 'Change'
            # process feature
            feature = self.df.iloc[pt:pt + WIN_LEN, 0:6].values
            feature = feature.T

            self.features.append(feature)
            self.class_labels.append(self.class_to_number(label))

            pt += int(WIN_LEN * OVERLAPPING)

        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)

        self.dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.features).float(),
                                                       torch.from_numpy(self.class_labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.dataset[idx]

    def class_to_number(self, label):
        dic = {'Not Change': 0,
               'Change': 1,
               }
        return dic[label]

if __name__ == '__main__':
    dataset = BeatChange_Dataset()