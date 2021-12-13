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

def merge_data(args):
    '''
    Merge all the processed data whose folder contains conf.args.regex and saves the combined file into dataset directory.
    '''
    raw_data_path = os.getcwd() + "/processed"
    directories = []
    pattern_of_path = args.regex
    pattern_of_path = re.compile(pattern_of_path)

    # search files and add if the regex matches.
    for (root, dirs, files) in os.walk(raw_data_path):
        for file in files:
            if not pattern_of_path.match(file): continue
            fdir = root + "/" + file
            directories.append(fdir)

    # read and combine data.
    # save the combined data into the "dataset" directory with filename specified in args.dataset_name
    if len(directories) == 0:
        print("no matching file for " + args.regex)
        return

    combined_csv_data = pd.concat([pd.read_csv(f) for f in directories]).iloc[:, 1:] # iloc to ignore unnecessary index
    combined_csv_data = combined_csv_data.reset_index(drop=True)
    print(combined_csv_data)

    # save as csv
    dirname = datetime.now().strftime('%Y%m%d') + "_" + args.dataset_name
    save_dir = str(Path().absolute()) + "/dataset/" + dirname
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    combined_csv_data.to_csv(save_dir + "/accgyro.csv", sep=',', na_rep='NaN', index=False)

def parse_arguments(argv):
    '''
    Parse a command line.
    '''
    parser = argparse.ArgumentParser()

    ### MANDATORY ###
    parser.add_argument('--regex', type=str, default='',
                        help='keyword of the files to combine.')
    parser.add_argument('--dataset_name', type=str, default='',
                        help='Name of the combined data.')

    return parser.parse_args()


if __name__=='__main__':
    args = parse_arguments(sys.argv[1:])
    merge_data(args)
    # python merge_data.py --regex .*debug.* --dataset_name debug