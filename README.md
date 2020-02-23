# LCN (Lightweight Crowd-corunting Net) by using NWPU-Crowd dataset 

---

This repo is based on [NWPU-Crowd-Sample-Code](https://github.com/gjy3035/NWPU-Crowd-Sample-Code). 

# Getting Started

## Preparation
- Prerequisites
  - Python 3.x
  - Pytorch 1.x: http://pytorch.org .
  - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.

- Installation
  - Clone this repo:
    ```
    git clone https://github.com/sailyung/crowd_counting.git
    ```
  
- Data Preparation
  - Download NWPU-Crowd dataset from this [link](https://mailnwpueducn-my.sharepoint.com/personal/gjy3035_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FNWPU%2DCrowd&originalPath=aHR0cHM6Ly9tYWlsbndwdWVkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2dqeTMwMzVfbWFpbF9ud3B1X2VkdV9jbi9Fc3ViTXA0OHd3SkRpSDBZbFQ4Mk5ZWUJtWTlMMHMtRnByckJjb2FBSmtJMXJ3P3J0aW1lPWdxTkxjV0dTMTBn). 
  - Unzip ```*zip``` files in turns and place ```images_part*``` into a folder. Finally, the folder tree is below:
  ```
    -- NWPU-Crowd
        |-- images
        |   |-- 0001.jpg
        |   |-- 0002.jpg
        |   |-- ...
        |   |-- 5109.jpg
        |-- jsons
        |   |-- 0001.json
        |   |-- 0002.json
        |   |-- ...
        |   |-- 5109.json
        |-- mats
        |   |-- 0001.mat
        |   |-- 0002.mat
        |   |-- ...
        |   |-- 5109.mat
        |-- train.txt
        |-- val.txt
        |-- test.txt
        |-- readme.md
  ```
  - Run ```./datasets/prepare_NWPU.m``` using [Matlab](https://www.mathworks.com/). 
  - Modify ```__C_NWPU.DATA_PATH``` in ```./datasets/setting/NWPU.py``` with the path of your processed data.


## Training

- Set the parameters in ```config.py``` and ```./datasets/setting/NWPU.py``` (if you want to reproduce our results, you are recommended to use our parameters in ```./saved_exp_para```).
- run ```python train.py```.
- run ```tensorboard --logdir=exp --port=6006```.

## Testing

We only provide an example to forward the model on the test set. You may need to modify it to test your models.

- Modify some key parameters in ```test.py```: 
  - Line 32: ```LOG_PARA```, the same as ```__C_NWPU.LOG_PARA``` in ```./datasets/setting/NWPU.py```.
  - Line 34: ```dataRoot```, the same as ```__C_NWPU.DATA_PATH``` in ```./datasets/setting/NWPU.py```.
  - Line 36: ```model_path```.  
  - Line 48: GPU Id and Model Name. 
- Run ```python test.py```.

# Performance on the validation set

The overall results on val set:

|   Method   |  O_MAE  |  O_MSE  |  O_NAE  |  Avg.MAE[S]  |  Avg.MAE[L]  | 
|------------|-------|-------|--------|--------|--------|
| MCNN [1]   | 232.5 | 714.6 | 1.063 | 1171.9 | 220.9 |
| SFCN+ [2]  |  105.7| 424.1| 0.254 | 712.7 | 106.8 | 
| LCNet | 233.097 | 802.123 | 0.848 | 1503.09 | 218.682 |

O_MAE: Mean Absolute Error (MAE) on overall testing images.
O_MSE: Mean Squared Error (MAE) on overall testing images.
O_MAE: Normalized Absolute Error (NAE) on overall testing images.
Avg.MAE[S]: The test set is divided into five scene levels according to the numbers of people in an image, and avg. MAE[S] represents the average MAE of the five scene levels.
Avg.MAE[L]: The test set is divided into four luminance levels according to the luminance calculated in an image, and avg. MAE[S] represents the average MAE of the four luminance levels.

About the leaderboard on the test set, please visit [Crowd benchmark](https://crowdbenchmark.com/nwpucrowd.html).  

## References

1. Single-Image Crowd Counting via Multi-Column Convolutional Neural Network, CPVR, 2016.
2. Learning from Synthetic Data for Crowd Counting in the Wild, CVPR, 2019.


# Evaluation Scheme 

The Evaluation Python Code of the ```crowdbenchmark.com``` is shown in ```./misc/evaluation_code.py```, which is similar to our validation code in ```trainer.py```. 
