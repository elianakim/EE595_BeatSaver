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

## Dataset

### Download Preprocessed Dataset
We provide our preprocessed datasets in a csv format [[here](https://drive.google.com/drive/folders/1Fu6KItvxJ2z-gB8PKpuzB7N6gX8Wo0NB?usp=sharing)]
* ichar_minmax_scaling_all.csv: minmax-scaled (0-1) ICHAR dataset used in our experiment  
* icsr_minmax_scaling_all.csv: minmax-scaled (0-1) ICSR dataset used in our experiment
* hhar_minmax_scaling_all.csv: minmax-scaled (0-1) HHAR dataset used in our experiment
* dsa_minmax_scaling_all.csv: minmax-scaled (0-1) DSA dataset used in our experiment
* ichar_original_all.csv: ICHAR dataset before scaling (for different purposes)
* icsr_original_all.csv: ICSR dataset before scaling (for different purposes)

To run our codes, you first need to download at least one of the datasets. After that, make a directory for the datasets:

    $cd .               #project root
    $mkdir dataset
and locate them in the `./dataset/` directory.

### Create Your Own Dataset

## How to Run
Please refer to the `./run.sh` file. It contains running commands for all methods, datasets, and number of shots.

If you want to verify your installation, you can do simply with the following command:
    
    $python main.py --dataset metasense_activity_scaled --method MetaSense --tgt PH0007-jskim --epoch 200 --log_suffix run_test_5shot_0.1  --src rest --train 1 --model MetaSense_Activity_model --nshot 5 --lr 0.1 

CAUTION: For the *TrC* method, it requires a pre-trained *Src* model. Please make sure you have trained a Src model with the same `--log_suffix` argument before training a TrC instance. 
