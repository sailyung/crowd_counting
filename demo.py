from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
import numpy as np
from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
import time

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

model_path = 'all_ep_608_mae_215.3_mse_831.6_nae_1.009.pth'

                     
def re_name_weight(weight_dict):
    #wts = torch.load('xxx.pth')
    new_wts = {}
    for i_key in weight_dict.keys():
        new_key= i_key.replace('module.','')
        print (new_key)
        new_wts[new_key] = weight_dict[i_key]
    return new_wts

def my_demo(file_list, model_path):
    Net_OK = ['Res101_SFCN', 'LCN']
    if(cfg.NET not in Net_OK):
        print('net is not Res101_SFCN  demo not work')
        return
    net = CrowdCounter(cfg.GPU_ID,cfg.NET)

    new_weight_dict = torch.load(model_path)
    if(cfg.GPU_ID == [0]):
        new_weight_dict = re_name_weight(new_weight_dict)
    net.load_state_dict(new_weight_dict)
    net.cuda()
    net.eval()
    print('net eval is ok=================')

    f1 = plt.figure(1)
    for filename in file_list:
        print( filename )
        img = Image.open(filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)
        with torch.no_grad():
            img = Variable(img[None,:,:,:]).cuda()
            start = time.time()
            for i in range(1000):
                pred_map = net.test_forward(img)
            pred_map.cpu()
            end = time.time()
            density_pre = pred_map.squeeze().cpu().numpy() / 100.
            num_people = int(np.sum(density_pre))
            print('in this picture,there are ',num_people,' people')
            print('Do once forward need {:.3f}ms '.format((end-start)*1000/100.0))


if __name__ == '__main__':
    file_list = ['./A/people15.jpg']
    my_demo(file_list, model_path)

