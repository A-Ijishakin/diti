import torch 
import torchvision.transforms as tfs 
import numpy as np
from glob import glob
from PIL import Image 
import os 
from tqdm import tqdm 
import pandas as pd


class ImgLoader(torch.utils.data.Dataset):
    def __init__(self, dataset, h_creation=False, 
                 size=512, 
                 device_num=0):

        self.celeba = '/home/rmapaij/HSpace-SAEs/datasets/celeba/img_align_celeba'
        self.ffhq = '/home/rmapaij/HSpace-SAEs/datasets/FFHQ/images'

        self.size = size
        path = self.celeba if dataset == 'celeba' else self.ffhq
        imgs = f'{path}/*'  

        self.image_list = sorted(glob(imgs))
        
        
        self.h_creation = h_creation 
        
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        
        self.device_num = device_num 
        
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx, native=True, xs=None): 
        x = Image.open(self.image_list[idx]) if native else Image.open(xs[idx])
        
        # Crop the center of the image
        w, h = x.size 
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((self.size, self.size))      
                
        if self.transform is not None:
            x = self.transform(x) 
            
        path = self.image_list[idx] if self.h_creation else ''
        
        
        return {'img': x.to(device=f'cuda:{self.device_num}', dtype=torch.float32), 
                'index' : idx, 'path': path}   
        