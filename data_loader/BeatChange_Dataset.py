import os

import torch
import pandas as pd
import time
import numpy as np
import sys
import random
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

sys.path.append('..')
import conf

opt = conf.BeatChange_Opt
WIN_LEN = opt['seq_len']
RAW = opt['raw']    # raw data OR feature
SCALE = opt['scale']
OVERLAPPING = opt['overlapratio'] # overlapping window


class BeatChange_Dataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/20211211_meta/accgyro.csv'):
        print('Loading data...')

        st = time.time()

        self.features = None
        self.data_per_class = None
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
        self.data_per_class = {'Not Change': [], '3beats_1': [], '3beats_2': [], '3beats_3': [], 'Change': []}
        self.class_labels = []
        pt = 0
        cbin = {'Not Change': 0, '3beats_1': 0, '3beats_2': 0, '3beats_3': 0, 'Change': 0}
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
                    label = l
                    # label = 'Change'
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

                feature = np.zeros((WIN_LEN, 12)) #[None] * WIN_LEN
                feature_label = np.zeros((WIN_LEN, 1))

                flag = False
                cls = None
                for i in range(WIN_LEN):
                    if raw_label[i][0] != 'None':
                        flag= True
                        cls = self.class_to_number(raw_label[i][0])
                if flag:
                    feature_label = np.full((WIN_LEN, 1), cls)
                else:
                    feature_label = np.zeros((WIN_LEN, 1))

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
            feature_label = feature_label.T

            if conf.args.downsample:
                self.data_per_class[label].append(feature)
            else:
                self.features.append(feature)
                #self.class_labels.append(self.class_to_number(label))
                self.class_labels.append(feature_label)

            pt += int(WIN_LEN * OVERLAPPING)

        print(cbin) # print data statistics

        if conf.args.downsample:
            # get the minimum number (nonzero) of data, down-sample the dataset
            minnumdata = np.inf
            for c in list(cbin.keys()):
                if minnumdata > cbin[c] > 0:
                   minnumdata = cbin[c]
            for c in list(self.data_per_class.keys()):
                # randomly sample data
                if cbin[c] > minnumdata:
                    sampled = random.sample(self.data_per_class[c], minnumdata)
                    self.features.extend(sampled)
                    for i in range(len(sampled)):
                        self.class_labels.append(self.class_to_number(c))
                else:
                    self.features.extend(self.data_per_class[c])
                    for i in range(len(self.data_per_class[c])):
                        self.class_labels.append(self.class_to_number(c))

        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)

        self.dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.features).float(),
                                                       torch.from_numpy(self.class_labels))

        # Export csv
        '''
        for i in range(self.features.shape[0]):
            if i is 0:
                df_feature = pd.DataFrame(self.features[i].T)
            else:
                df_feature = df_feature.append(pd.DataFrame(self.features[i].T))
        #df_feature = pd.DataFrame(self.features.T)
        df_feature.columns = ['acc_x', 'acc_y','acc_z','acc_mean','acc_std','acc_var',
                              'gyro_x', 'gyro_y', 'gyro_z', 'gyro_mean', 'gyro_std', 'gyro_var']
        df_label = pd.DataFrame(np.squeeze(self.class_labels).flatten())
        df_label.columns = ['label']
        

        # df = pd.concat([df_feature, df_label], axis = 1, ignore_index=True)
        #df_feature["label"] = df_label
        df = df_feature.reset_index(drop=True)
        df['label'] = df_label
        #df = df_feature
        
        # make csv file
        df.to_csv('../dataset/20211211_meta/feature_labeled.csv', index=False)
        '''



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self.dataset[idx]

    def class_to_number(self, label):
        dic = {'Not Change': 0,
               '3beats_1': 1,
               '3beats_2': 2,
               '3beats_3':3
               # 'Change': 1,
               }
        return dic[label]

if __name__ == '__main__':

    dataset = BeatChange_Dataset()