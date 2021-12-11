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
RAW = opt['raw']    # raw data OR feature
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

            if RAW:
                # process feature
                feature = self.df.iloc[pt:pt + WIN_LEN, 0:6].values

            else:
                # process feature
                raw = self.df.iloc[pt:pt + WIN_LEN, 0:6].values
                feature = np.zeros((WIN_LEN, 12)) #[None] * WIN_LEN

                # Split raw data into acc and gyro
                raw_acc = raw[0:WIN_LEN, 0:3]
                raw_gyro = raw[0:WIN_LEN, 3:6]
                ## np.std(raw[0:WIN_LEN, 0]) # Calculate the std value of each axis within WIN_LEN

                for i in range(WIN_LEN):
                    acc_mean = np.mean(raw[i,0:3])
                    acc_std = np.std(raw[i,0:3])
                    acc_var = np.var(raw[i,0:3])

                    gyro_mean = np.mean(raw[i,3:6])
                    gyro_std = np.std(raw[i,3:6])
                    gyro_var = np.var(raw[i,3:6])

                    # arr includes raw_acc and feature_acc
                    arr = np.append(np.array(raw_acc[i]),np.array([acc_mean, acc_std, acc_var]))
                    # arr includes raw_acc, feature_acc, raw_gyro, and feature_gyro
                    arr = np.append(arr,np.append(np.array(raw_gyro[i]), np.array([gyro_mean, gyro_std, gyro_var])))

                    feature[i,0:12] = arr


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