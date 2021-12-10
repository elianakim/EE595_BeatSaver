import argparse
import sys
import os
import re
import glob
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

import conf

'''
Functions
'''
def label_is_valid(label):
    '''
    Checks the file contains synchronization information.
    Checks all the behaviors are labeled without errors.
    :param label    pandas dataframe
    :return True if the label file is valid
    '''
    # Checks the file contains synchronization information.
    sync_info = label["Behavior"].iloc[0]
    if sync_info != "Sync_end":
        print("The label does not contain synchronization details.")
        return False
    # Checks all the behaviors are labeled without errors.
    num_data = label.shape[0]
    behaviors = label["Behavior"]
    seq = None
    if conf.args.beats == 2:
        seq = ['2beats_1', '2beats_2']
    elif conf.args.beats == 3:
        seq = ['3beats_1', '3beats_2', '3beats_3']
    elif conf.args.beats == 4:
        seq = ['4beats_1', '4beats_2', '4beats_3', '4beats_4']

    debug = []
    prev = None
    reset_prev = True
    for i in range(2, num_data):
        behavior = behaviors.iloc[i]
        if reset_prev:
            prev = seq.index(behavior)
            reset_prev = False
            continue
        prev += 1
        if prev >= len(seq): prev = 0
        if behavior != seq[prev]:
            debug.append(i)
            reset_prev = True
    if len(debug) > 0:
        for i in debug:
            print("Check timestamp at index %d" % i)
        return False
    return True

def read_label():
    '''
    Reads the label file, drops unnecessary columns, and returns label dataframe.
    :return: pandas dataframe containing label information
    '''
    label = pd.read_csv(conf.args.label_filepath) # csv file format
    label = label.drop(
        columns=["Observation id", "Observation date", "Description", "Media file", "Total length", "FPS", "Subject",
                 "Behavioral category", "Modifiers", "Behavior type", "Stop (s)", "Duration (s)", "Comment start", "Comment stop"])
    assert(label_is_valid(label))
    return label

def sync_label(label):
    '''
    Sync the time of label with IMU data log.
    :param label: unsynced label (pandas dataframe)
    :return: synced label (pandas dataframe)
    '''
    start_time = label["Start (s)"].iloc[0]
    label["Start (s)"] -= start_time
    return label

def get_imu_sync_data():
    '''
    Reads sync data for imu values using accelerometer data. Saves values to conf.IMU_Process
    Saves:   index of the last synchronization pulse
             type: int
    For sensor data where the latter parts are lost, es and ee = 0.
    :param dir: directory of accelerometer data file. (for convenience, we use only acc data to find the sync info)
    '''
    sync_df = pd.read_csv(conf.args.sync_filepath)
    paths = sync_df["imu_filepath"].tolist() # keyerror when directly apply loc to the dataframe
    conf.IMU_Process['sync'] = sync_df.iloc[paths.index(conf.args.imu_filepath)].tolist()[1]

def preprocess_imu():
    '''
    Reads the data, drop unnecessary columns & rows, and sync data with labels.
    :return: pandas dataframe of accelerometer and gyroscope.
    '''
    df = pd.read_csv(conf.args.imu_filepath)
    df = df.iloc[conf.IMU_Process['sync']:]
    df = df.reset_index(drop=True)
    return df

def add_labels(df, l_df):
    '''
    Using l_df, label each data instance in df.
    Since Arduino does not record the timestamp, we depend on the sampling rate.
    Also, discard data after the label ended.
    :param df: dataframe of acc+gyro dataset
    :param l_df: dataframe of label, consisting of each behavior and its timestamp.
    :return: dataframe with labels.
    '''
    label_dict = {} # key: index of the data to be labeled. value: the label (ex. 3beats_1)
    first_label = -1
    last_label = -1
    for i in range(2, l_df.shape[0]):
        # estimate the index by the elapsed time, using the sampling rate.
        # the sampling rate of the accelerometer and the gyroscope are the same. (thankfully!)
        timestamp = l_df["Start (s)"].iloc[i]
        idx = int(conf.IMU_Process["acc_sampling_rate"] * timestamp)
        label_dict[idx] = l_df["Behavior"].iloc[i]
        if i == 2:
            first_label = idx
        if i == l_df.shape[0] - 1:
            last_label = idx

    labels = []
    discard_before_this_idx = -1
    discard_after_this_idx = -1
    for idx in range(df.shape[0]):
        if idx not in list(label_dict.keys()):
            labels.append("None")
        else:
            labels.append(label_dict[idx])
            if idx == first_label:
                discard_before_this_idx = idx
            if idx == last_label:
                discard_after_this_idx = idx

    df['beat_change'] = labels
    df = df.iloc[discard_before_this_idx:discard_after_this_idx]
    df = df.reset_index(drop=True)

    # add another label indicating what kind of beat it is
    beat_labels = [conf.args.beats for i in range(df.shape[0])]
    df['beat'] = beat_labels

    return df

def main():
    ''' data process '''

    '''
    Pre-process label.
    '''
    l_df = sync_label(read_label()) # read and sync label

    '''
    Pre-process IMU data. 
    '''
    get_imu_sync_data() # get synchronization data (saved in conf.Temp)
    df = preprocess_imu() # preprocess data

    '''
    Add labels IMU data according to using l_df
    '''
    df = add_labels(df, l_df)

    '''
    save as csv
    '''
    dirname = datetime.now().strftime('%Y%m%d') + "_" + conf.args.output_suffix # %Y%m%d-%H:%M:%S
    save_dir = str(Path().absolute()) + "/processed/" + dirname
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df.to_csv(save_dir + "/" + dirname + ".csv", sep=',', na_rep='NaN')

def parse_arguments(argv):
    """
    Parse a command line.
    """
    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    ### MANDATORY ###
    parser.add_argument('--imu_filepath', type=str, default='',
                        help='Path to the file that contains IMU sensor logs.')
    parser.add_argument('--label_filepath', type=str, default='',
                        help='Path to the activity labels.')
    parser.add_argument('--sync_filepath', type=str, default='',
                        help='Path to the file that includes start and end time of synchronization phase.')
    parser.add_argument('--output_suffix', type=str, default='debug',
                        help='Suffix of output file path')
    parser.add_argument('--beats', type=int, default=2,
                        help='Beat type. 2, 3, or 4')
    parser.add_argument('--dynamics', type=str, default='f',
                        help='dynamic of the beat. "f" if forte, "p" if piano.')

    ### OPTIONAL ###
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    return parser.parse_args()

def set_seed():
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)

if __name__=='__main__':
    conf.args = parse_arguments(sys.argv[1:])
    set_seed()
    main()
    # example command
    # python data_process.py --imu_filepath /mnt/sting/yewon/EE595_BeatSaver/rawdata/data/f3_yewon.csv --label_filepath /mnt/sting/yewon/EE595_BeatSaver/rawdata/label/f3_yewon_label.csv --sync_filepath /mnt/sting/yewon/EE595_BeatSaver/rawdata/sync.csv --output_suffix debug --beats 3 --dynamics f
