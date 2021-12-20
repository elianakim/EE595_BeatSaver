# EE595 BeatSaver: Conducting Gestures to Metronome Beats
This is the PyTorch Implementation of BeatSaver. 

## Tested Environment
We collected our data under this environment. 
- Arduino Nano 33 BLE Sense

We tested our codes under this environment.
- OS: Ubuntu 18.04.5 LTS
- GPU: TITAN RTX 24G
- GPU Driver Version: 470.74
- CUDA Version: 11.4

## Installation Guide
1. Download or clone our repository
2. Set up a python environment using conda (explained in *Python Environment*)
3. Download preprocessed dataset or create your own dataset (explained in *Dataset*)
4. Run the code

## Python Environment
We use [Conda environment](https://docs.conda.io/)
You can get conda by installing [Anaconda](https://www.anaconda.com/) first.

We share our python environment that contains all required python packages. Please refer to the './beatsaver.yml' file.

You can import our environment using conda:

    conda env create -f beatsaver.yml -n beatsaver
   
Reference: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

## Quick Guide for Demo (for TAs)
We first provide the quick guide for course TAs to evaluate our project, since the full reproduction of our project (from model training) takes more than several hours. 

### Evaluating Demo



### Playing Demo


Below, we describe the details of the project, including guidelines to preprocess/use dataset, train models, and producing beats outputs for further references.

## Dataset

### Use Our Preprocessed Dataset
We provide our preprocessed datasets in a csv format [[here](https://github.com/elianakim/EE595_BeatSaver/tree/main/dataset)]
IMU data are named as "accgyro.csv".
* train_all: train dataset including all time signatures for training Time Signature Classifier
* train_forte_2beats: train dataset for time signature of 2/4 and dynamics of forte.
* train_forte_3beats: train dataset for time signature of 3/4 and dynamics of forte.
* train_forte_4beats: train dataset for time signature of 4/4 and dynamics of forte.
* train_piano_2beats: train dataset for time signature of 2/4 and dynamics of piano.
* train_piano_3beats: train dataset for time signature of 3/4 and dynamics of piano.
* train_piano_4beats: train dataset for time signature of 4/4 and dynamics of piano.
* demo_2beats: test dataset for time signature of 2/4 with varying dynamics.
* demo_3beats: test dataset for time signature of 3/4 with varying dynamics.
* demo_4beats: test dataset for time signature of 4/4 with varying dynamics.

### Create Your Own Dataset

Prepare raw data and label in the *rawdata* folder. 

Open [[this link](https://colab.research.google.com/drive/11zmhghSF33tl8GBEkA5091RSE1tDs3Ov?usp=sharing)] to synchronize the data. To be specific, find the index of the raw data where the synchronization pulses end. Write the information with the path to raw data in sync.csv in the *rawdata* folder.

You can preprocess data with the following command (in this case, 3/4 time signature with dynamics of forte):
    
    $python data_process.py --imu_filepath path-to-raw-data --label_filepath path-to-label --sync_filepath path-to-sync.csv --output_suffix create_dataset --beats 3 --dynamics f

Merge the preprocessed data into dataset. 

    $python merge_data.py --regex .*create_dataset.* --dataset_name example_dataset

## How to Run
Please refer to the `./run.sh` file. It contains running commands for all methods, datasets, and number of shots.

If you want to verify your installation, you can do simply with the following command:
    
    $python main.py --dataset metasense_activity_scaled --method MetaSense --tgt PH0007-jskim --epoch 200 --log_suffix run_test_5shot_0.1  --src rest --train 1 --model MetaSense_Activity_model --nshot 5 --lr 0.1 

CAUTION: For the *TrC* method, it requires a pre-trained *Src* model. Please make sure you have trained a Src model with the same `--log_suffix` argument before training a TrC instance. 

## Misc.
### 1. Checkpoint
The code will save the validation best model (i.e., checkpoint) in the logging directory.

For instance, `LOG_DIR/cp/cp_best.pth.tar`

### 2. Result
After the training, it will generate a result.txt file that provides the validation best accuracy and the test accuracy.
 
For instance, `LOG_DIR/result.txt`

## Acknowledgement
We imported the format of the project from [MetaSense_public](https://github.com/TaesikGong/MetaSense_public).
