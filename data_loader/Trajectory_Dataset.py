import os

import torch
import pandas as pd
import time
import numpy as np
import sys
import random
import math

sys.path.append('..')
import conf

opt = conf.BeatChange_Opt

''' 일단 LABEL_INDEX = 13으로 함. '''
opt['raw'] = False
if opt['raw']:
    LABEL_INDEX = 6     # when using only raw data
else:
    LABEL_INDEX = 13    # when using both raw data and feature


class Trajectory_Dataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/20211211_f3_re_yewon/feature_labeled.csv'):
        print('Loading data...')

        st = time.time()

        #self.features = None
        #self.data_per_class = None
        self.class_labels = None
        #self.dataset = None
        #self.classes = conf.BeatChange_Opt['classes']
        self.beat_timestamp = None

        self.df = pd.read_csv(file)
        self.generated_df = None
        self.extract_beats()
        self.preprocessing()
        ppt = time.time()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def extract_beats(self):
        ''' Extract beats from the original raw data so that we can have test data '''

        ''' Label Processing '''
        # decide label

        labels = self.df.iloc[:]['label']
        '''
        # label-class to number
        for l in labels:
            if l != 'None':
                label = l
            else:
                label = 0

            self.class_labels.append(self.class_to_number(label))
        '''
        self.class_labels = pd.DataFrame(labels)

        ''' Generate Beat Timestamp(index) Sequence '''
        df = self.df
        beat_timestamp = []
        class_labels = self.class_labels
        for i in range(len(df)):
            # if ith data is beat_change, store the index i in beat_timestamp
            if class_labels['label'][i] > 0:
                beat_timestamp.append(i)

        beat_timestamp_df = pd.DataFrame(beat_timestamp)
        # beat_timestamp_df.to_csv('../dataset/20211211_f3_re_yewon/beat_timestamp.csv')
        self.beat_timestamp = beat_timestamp_df



    def preprocessing(self):
        ''' Generate trajectory data between beats'''
        '''
            # self.df               # dataframe
            # self.beat_timestamp   # beat timestamp(index)
            # self.class_labels     # class labels
        '''
        self.features = []

        df = self.df
        generated_df = pd.DataFrame()
        # microTime_old = time.time() *1000000

        idx = 0
        for i in range(len(self.beat_timestamp)):
            timestamp = self.beat_timestamp[0][i]
            if idx is 0: # if idx is 0, it means that there is no previous beat
                idx = timestamp
                microTime_old = time.time()*1000000
                # microTime_old = df['timestamp_ms'][timestamp]
                rotate_x = 0
                rotate_y = 0
                rotate_z = 0
                vx = 0
                vy = 0
                vz = 0
                x = 0
                y = 0
                z = 0
                continue
            previous_timestamp = idx
                # Initialize acc_offset
            ax_offset = df['acc_x'][previous_timestamp]
            ay_offset = df['acc_y'][previous_timestamp]
            az_offset = df['acc_z'][previous_timestamp]

            # microTime_old = df['timestamp_ms'][previous_timestamp-1]

            # Calculate trajectory
            for j in range(previous_timestamp+1, timestamp):
                microTime = time.time() * 1000000
                # microTime = df['timestamp_ms'][j]
                microTime_delta = microTime - microTime_old
                microTime_old = microTime

                ''' Read gyroscope and compute the angle '''
                gx = -df['gyro_x'][j] # X axis reverse
                gy = df['gyro_y'][j]
                gz = df['gyro_z'][j]
                rotate_x = rotate_x + gx * microTime_delta / 1000000 * (2*math.pi/360);
                rotate_y = rotate_y + gy * microTime_delta / 1000000 * (2*math.pi/360);
                rotate_z = rotate_z + gz * microTime_delta / 1000000 * (2*math.pi/360);

                ''' Read the acceleration (relative to the device's axis) '''
                ax = -df['acc_x'][j] # X axis reverse
                ay = df['acc_y'][j]
                az = df['acc_z'][j]

                ''' Remove roll '''
                ax_tmp = ax
                ay_tmp = ay * math.cos(rotate_x) - az * math.sin(rotate_x)
                az_tmp = az * math.sin(rotate_x) + ax * math.cos(rotate_x)
                ax = ax_tmp
                ay = ay_tmp
                az = az_tmp

                ''' Remove pitch '''
                ax_tmp = ax * math.cos(rotate_y) + az * math.sin(rotate_y)
                ay_tmp = ay
                az_tmp = - ax * math.sin(rotate_y) + az * math.cos(rotate_y)
                ax = ax_tmp
                ay = ay_tmp
                az = az_tmp

                ''' Remove yaw '''
                ax_tmp = ax * math.cos(rotate_z) - ay * math.sin(rotate_z)
                ay_tmp = ax * math.sin(rotate_z) + ay * math.sin(rotate_z)
                az_tmp = az
                ax = ax_tmp
                ay = ay_tmp
                az = az_tmp

                ''' Remove acceleration offset '''
                ax = ax - ax_offset
                ay = ay - ay_offset
                az = az - az_offset

                ''' Now we have acc with respect to the absolute(world) axis
                    # Double integral to compute location '''
                vx = vx + 9.8 * ax * microTime_delta/1000000
                vy = vy + 9.8 * ay * microTime_delta/1000000
                vz = vz + 9.8 * az * microTime_delta/1000000

                x = x + vx * microTime_delta / 1000000
                y = y + vy * microTime_delta / 1000000
                z = z + vz * microTime_delta / 1000000

                x_sqr = x*x*10000
                y_sqr = y*y*10000
                z_sqr = z*z*10000
                dist = math.sqrt(x_sqr+y_sqr+z_sqr)
                #TODO: 각 beat_timestamp마다 dist저장하

                row = df.iloc[[j]]
                row['x'] = x
                row['y'] = y
                row['z'] = z
                #row['microTime'] = microTime
                row['microTime'] = j
                generated_df = generated_df.append(row)
            idx = timestamp
        generated_df.to_csv('../dataset/20211211_f3_re_yewon/beat_timestamp_trajectory.csv')


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
               '3beats_3': 3
               #'Change': 1,
               }
        return dic[label]


if __name__ == '__main__':
    dataset = Trajectory_Dataset()