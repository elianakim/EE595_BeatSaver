import os

import torch
import pandas as pd
import time
import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler

sys.path.append('..')
import conf

opt = conf.args.opt
WIN_LEN = opt['seq_len']
RAW = opt['raw'] # raw data or feature
SCALE = opt['scale']
OVERLAPPING = opt['overlap_ratio'] # overlapping window

class BeatType_Dataset(torch.utils.data.Dataset):

    def __init__(self, file=opt['file_path']):
        print('Loading data...')

        st = time.time()

        self.features = None
        self.class_labels = None
        self.dataset = None
        self.classes = opt['classes']

        self.df = pd.read_csv(file)
        self.preprocessing()
        ppt = time.time()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):
        self.features = []
        self.class_labels = []
        cbin = {'2': 0, '3': 0, '4': 0}

        pt = 0

        while pt + WIN_LEN <= len(self.df):
            bt = time.time()
            # decide label
            labels = self.df.iloc[pt : pt+WIN_LEN, opt['label_index']].values
            dynamics = self.df.iloc[pt : pt+WIN_LEN, opt['label2_index']].values
            domains = list(zip(labels, dynamics))

            # if the multiple domains are included in one window, skip the data.
            first_label = domains[0]
            if domains.count(first_label) < len(domains):
                pt += int(WIN_LEN * OVERLAPPING)
                continue
            else:
                label = str(first_label[0])
                cbin[label] += 1

            if RAW:
                # process feature
                feature = self.df.iloc[pt:pt + WIN_LEN, 0:6].values
                if SCALE:
                    scaler = StandardScaler()
                    feature_scaled = scaler.fit_transform(feature)
                    feature = feature_scaled
            else:
                # process feature
                raw = self.df.iloc[pt:pt + WIN_LEN, 0:6].values
                raw_label = self.df.iloc[pt:pt + WIN_LEN, 6:7].values

                feature = np.zeros((WIN_LEN, 12))  # [None] * WIN_LEN
                # feature_label = np.zeros((WIN_LEN, 1))
                #
                # flag = False
                # cls = None
                # for i in range(WIN_LEN):
                #     if raw_label[i][0] != 'None':
                #         flag = True
                #         cls = self.class_to_number(raw_label[i][0])
                # if flag:
                #     feature_label = np.full((WIN_LEN, 1), cls)
                # else:
                #     feature_label = np.zeros((WIN_LEN, 1))
                #
                # Split raw data into acc and gyro
                raw_acc = raw[0:WIN_LEN, 0:3]
                raw_gyro = raw[0:WIN_LEN, 3:6]
                ## np.std(raw[0:WIN_LEN, 0]) # Calculate the std value of each axis within WIN_LEN

                for i in range(WIN_LEN):
                    acc_mean = np.mean(raw[i, 0:3])
                    acc_std = np.std(raw[i, 0:3])
                    acc_var = np.var(raw[i, 0:3])

                    gyro_mean = np.mean(raw[i, 3:6])
                    gyro_std = np.std(raw[i, 3:6])
                    gyro_var = np.var(raw[i, 3:6])

                    # arr includes raw_acc and feature_acc
                    arr = np.append(np.array(raw_acc[i]), np.array([acc_mean, acc_std, acc_var]))
                    # arr includes raw_acc, feature_acc, raw_gyro, and feature_gyro
                    arr = np.append(arr, np.append(np.array(raw_gyro[i]), np.array([gyro_mean, gyro_std, gyro_var])))

                    feature[i, 0:12] = arr
                    '''
                    if raw_label[i][0] == 'None':
                        feature_label[i] = np.array(self.class_to_number('Not Change'))
                    else:
                        feature_label[i] = np.array(self.class_to_number(raw_label[i][0]))
                    '''

                if SCALE:
                    scaler = StandardScaler()
                    feature_scaled = scaler.fit_transform(feature)
                    feature = feature_scaled

            feature = feature.T
            #
            # if not RAW:
            #     feature_label = feature_label.T

            self.features.append(feature)
            self.class_labels.append(self.class_to_number(label))

            pt += int(WIN_LEN * OVERLAPPING)

        print(cbin)

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
        dic = {'2': 0,
               '3': 1,
               '4': 2,
               }
        return dic[label]

if __name__ == '__main__':

    dataset = BeatType_Dataset()
