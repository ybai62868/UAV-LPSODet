from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DACDataset(Dataset):
    def __init__(self, list_path, img_size, transform):
        with open(list_path, 'r') as fout:
            self.img_files = fout.readlines()
        self.label_path = [path.replace('.jpg', '.xml') for path in self.img_files]
        self.img_size = img_size
        self.transform = transform
        self.db = []


    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path)
        h, w, _ = img.shape
        img_resized = img.resize(self.img_size) 
        np_img = np.array(img_resized) / 255.0
        input_img = np.transpose(np_img, (2, 0, 1))
        input_img = torch.tensor(input_img, dtype=torch.float)
        




        

    def __len__():
        return len(self.img_files)