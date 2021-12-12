import os

import torch
import pandas as pd
import time
import numpy as np
import sys

sys.path.append('..')
import conf

from sklearn.decomposition import PCA
from sklearn import StandardScaler
import pyplot.matplotlib as plt

opt = conf.BeatType_Opt
WIN_LEN = opt['seq_len']
OVERLAPPING = opt['overlap_ratio'] # overlapping window

class PCA_Dataset(torch.utils.data.Dataset):

    def __init__(self, file='./20211210_meta/feature_labeled.csv'):
        print('Loading data...')

        st = time.time()

        self.features = None
        self.x = None
        self.y = None

        #self.dataset = None
        #self.classes = conf.BeatChange_Opt['classes']

        # load dataset into Pandas DataFrame
        self.df = pd.read_csv(file)
        self.preprocessing()
        self.plot_pca()
        ppt = time.time()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):

        df = self.df
        self.features = ['acc_x', 'acc_y', 'acc_z', 'acc_mean', 'acc_std', 'acc_var',
                         'gyro_x', 'gyro_y', 'gyro_z', 'gyro_mean', 'gyro_std', 'gyro_var']

        # Separate out the features
        x = df.loc[:, self.features].values

        # Separate out the label
        y = df.loc[:, ['label']].values

        # Standardize the features
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        self.x = x
        self.y = y

    def plot_pca(self):
        df = self.df

        pca = PCA(n_components = 2)
        PCs = pca.fit_transform(self.x) # Principal Components 1 and 2

        principal_df = pd.DataFrame(data = PCs,
                                   columns = ['PC1', 'PC2'])

        final_df = pd.concat([principal_df, df[['label']]], axis = 1)

        # Visualize 2D projection
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.



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
    dataset = PCA_Dataset()