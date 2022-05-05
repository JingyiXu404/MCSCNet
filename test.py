import os
import math
import time
import torch
import random
import matplotlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms
from train_npy import msdataset
from model import MCSCNet
from torch.utils.data import DataLoader

import freeze
import psnr
import cv2
import scipy.io as sio
from test_npy import datatest

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def data_process_npy(data_in,mean,batch):
    data_out=data_in.cpu().numpy().astype(np.float32)
    data_out=np.transpose(data_out,(0,2,3,1))
    data_out=np.squeeze(data_out)
    data_out=data_out+mean
    data=data_out*255.
    data_out[:,:,0]=data[:,:,2]
    data_out[:,:,2]=data[:,:,0]
    data_out[:,:,1]=data[:,:,1]
    return data_out

def save_mat(data_in):
    data_out=data_in.cpu().numpy()
    data_out=np.transpose(data_out,(0,2,3,1))
    return data_out

def test():
    dataset='BSD68'
    noise_level =15
    test_set = datatest(dataset=dataset,noise_level=noise_level)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    model = MSNet()
    model = model.cuda()
    lr_img = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
              'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'twentyone',
              'twentytwo', 'twentythree', 'twentyfour', 'twentyfive', 'twentysix', 'twentyseven', 'twentyeight',
              'twentynine', 'thirty', 'thirtyone', 'thirtytwo', 'thirtythree', 'thirtyfour', 'thirtyfive', 'thirtysix',
              'thirtyseven', 'thirtyeight', 'thirtynine', 'forty', 'fortyone', 'fortytwo', 'fortythree', 'fortyfour',
              'fortyfive', 'fortysix', 'fortyseven', 'fortyeight', 'fortynine', 'fifty', 'fiftyone', 'fiftytwo',
              'fiftythree', 'fiftyfour', 'fiftyfive', 'fiftysix', 'fiftyseven', 'fiftyeight', 'fiftynine', 'sixty',
              'sixtyone', 'sixtytwo', 'sixtythree', 'sixtyfour', 'sixtyfive', 'sixtysix', 'sixtyseven', 'sixtyeight',
              'sixtynine', 'seventy', 'seventyone', 'seventytwo', 'seventythree', 'seventyfour', 'seventyfive',
              'seventysix', 'seventyseven', 'seventyeight', 'seventynine', 'eighty', 'eightyone', 'eightytwo',
              'eightythree', 'eightyfour', 'eightyfive', 'eightysix', 'eightyseven', 'eightyeight', 'eightynine',
              'ninety', 'ninetyone', 'ninetytwo', 'ninetythree', 'ninetyfour', 'ninetyfive', 'ninetysix', 'ninetyseven',
              'ninetyeight', 'ninetynine', 'hundred']

    state = torch.load('model/N'+str(noise_level)+'/'+'best.pth') #The name of the pretrained model
    model.load_state_dict(state['model'])
    model.eval()

    dic = {}
    PSNR = []
    SSIM = []
    with torch.no_grad():

        for batch_test, (lr, hr) in enumerate(test_loader):
            hr = hr.float()
            lr = lr.float()

            hr = hr.cuda()
            lr = lr.cuda()

            sr,z1,z2,z3= model(lr)

            data_out = save_mat(sr)
            dic[lr_img[batch_test] + '_dn'] = data_out
            #
            sr = data_process_npy(sr, 0, batch_test)
            hr = data_process_npy(hr, 0, batch_test)

            PSNR.append(psnr.psnr(sr, hr))
            SSIM.append(psnr.SSIM(sr, hr))
        ave_psnr = np.mean(PSNR)
        ave_ssim = np.mean(SSIM)
        if not os.path.exists('result/N'+str(noise_level)+'/MAT/'):
            os.makedirs('result/N'+str(noise_level)+'/MAT/')

        sio.savemat('result/N'+str(noise_level)+'/MAT/'+dataset+'.mat', dic) #save denoised image as .mat

    return ave_psnr, ave_ssim


val_psnr = test()
print(val_psnr)