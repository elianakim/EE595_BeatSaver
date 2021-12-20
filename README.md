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

    $ conda env create -f environment.yml -n beatsaver
    
Activate the environment by the following command:

    $ conda activate beatsaver
   
Reference: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

## Quick Guide for Evaluation (for TAs)
We first provide the quick guide for course TAs to evaluate our project, since the full reproduction of our project (from model training) takes more than several hours. 


### Evaluating Demo

Instead of uploading all models, we provide two example model checkpoints for Time Signature Classifier and Beat Detector since the trained models are large in size. Please download the directory containing files and extract it in the project directory: [[download](https://drive.google.com/file/d/1HkcW_KWUMJodKP3V-QmiXAV9oRYl5dBx/view?usp=sharing)]

To test Time Signature Classifier, run the following code:

    $ python main.py --type beat_type --dataset beat_original --model beat_type_model --method Demo --load_checkpoint_path  ./checkpoints/time_signature_classifier/cp/ --demo_produce --feat_eng --load_demo_data_path ./dataset/demo_2beats/accgyro.csv
    $ python main.py --type beat_type --dataset beat_original --model beat_type_model --method Demo --load_checkpoint_path  ./checkpoints/time_signature_classifier/cp/ --demo_produce --feat_eng --load_demo_data_path ./dataset/demo_3beats/accgyro.csv
    $ python main.py --type beat_type --dataset beat_original --model beat_type_model --method Demo --load_checkpoint_path  ./checkpoints/time_signature_classifier/cp/ --demo_produce --feat_eng --load_demo_data_path ./dataset/demo_4beats/accgyro.csv
    
To test Beat Detector, run the following code:

    $ python main.py --type beat_change --dataset beat_original --model beat_change_model --method Demo --load_checkpoint_path ./checkpoints/beat_detector/cp/ --demo_produce --feat_eng --load_demo_data_path ./dataset/demo_3beats/accgyro.csv --beat_type 3

### Playing Demo

To run the demo, run the following codes:

    $ python demo.py --beat 2
    $ python demo.py --beat 3
    $ python demo.py --beat 4

Below, we describe the details of the project, including guidelines to preprocess/use dataset, train models, and producing beats outputs for further references.

## Dataset

### Use Our Preprocessed Dataset
We provide our preprocessed datasets in a csv format [[here](https://github.com/elianakim/EE595_BeatSaver/tree/main/dataset)]. 
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

## Training the Models
Please refer to the `./script/train_models.sh` file. It contains running commands for various hyperparameters.

Run the following command to train models. 

    $ . script/train_models.sh
    
Using the models, you can produce the demo results and play demo as described in the quick guide for evaluation.

## Misc.
### 1. Checkpoint
The code will save the validation best model (i.e., checkpoint) in the logging directory.

For instance, `LOG_DIR/cp/cp_best.pth.tar`

### 2. Result
After the training, it will generate a result.txt file that provides the validation best accuracy and the test accuracy.
 
For instance, `LOG_DIR/result.txt`

## Acknowledgement
We imported the format of the project from [MetaSense_public](https://github.com/TaesikGong/MetaSense_public).
