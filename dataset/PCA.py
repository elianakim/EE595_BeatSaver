import os

import torch
import pandas as pd
import time
import numpy as np
import sys

sys.path.append('..')
import conf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

opt = conf.BeatType_Opt
WIN_LEN = opt['seq_len']
OVERLAPPING = opt['overlap_ratio'] # overlapping window

class PCA_Dataset(torch.utils.data.Dataset):

    def __init__(self, file='./20211211_meta/feature_labeled.csv'):
        print('Loading data...')

        st = time.time()

        self.features = None
        self.x = None
        self.y = None


        # Load dataset into Pandas DataFrame
        self.df = pd.read_csv(file)
        # Process dataset to plot PCA
        self.preprocessing()
        # Plot PCA
        self.plot_pca()
        ppt = time.time()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):

        df = self.df
        '''
        self.features = ['acc_x', 'acc_y', 'acc_z', 'acc_mean', 'acc_std', 'acc_var',
                         'gyro_x', 'gyro_y', 'gyro_z', 'gyro_mean', 'gyro_std', 'gyro_var']
        '''
        self.features = ['acc_mean', 'acc_std', 'acc_var',
                         'gyro_mean', 'gyro_std', 'gyro_']
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
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)

        labels = [0, 1, 2, 3]
        colors = ['r', 'b', 'g', 'y']
        for label, color in zip(labels, colors):
            indices_to_keep = final_df['label'] == label
            ax.scatter(final_df.loc[indices_to_keep, 'PC1'],
                       final_df.loc[indices_to_keep, 'PC2'],
                       c = color,
                       s = 50,
                       alpha = 0.2)
        #ax.legend(labels)
        ax.legend(['None', '3/4 beat 1', '3/4 beat 2', '3/4 beat3'])
        ax.grid()
        plt.show()



if __name__ == '__main__':
    dataset = PCA_Dataset()