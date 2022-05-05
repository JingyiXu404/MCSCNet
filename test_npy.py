import torch.utils.data as data
from glob import glob
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random
import os
import numpy as np
import torch
import cv2

class datatest(data.Dataset):
    def __init__(self,dataset,noise_level):
        super(datatest, self).__init__()
        filename = 'dataset/test_data' #The folder of your own .npy file containing groundtruth
        self.imgnames_HR=glob(os.path.join(filename,'gt_'+dataset,'*.npy'))
        self.imgnames_HR.sort()
        self.sigma = noise_level


    def __getitem__(self, item):

        self.gt = np.load(self.imgnames_HR[item],allow_pickle=True)
        self.gt = np.transpose(self.gt, (0, 3, 1, 2))
        self.gt = torch.from_numpy(self.gt)
        self.gt = np.squeeze(self.gt)
        self.noise = self.gt + (self.sigma / 255) * torch.randn_like(self.gt)
        return (self.noise, self.gt)

    def __len__(self):
        return len(self.imgnames_HR)




if __name__ == '__main__':
    dataset = datatest('BSD68',25)
    dataloader = data.DataLoader(dataset, batch_size=1)
    for b1, (img_L, img_H) in enumerate(dataloader):
        print(b1)
        print(img_L.shape, img_H.shape)