# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import  transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import numpy as np
import torch
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if is_train:
        root = os.path.join(args.root_path, 'train')
    else:
        root = os.path.join(args.root_path, 'test')

    if args.localization:
        dataset = gas_loc_dataset(root, transform, args.mask)
    else:
        dataset = gas_mapping_dataset(root, transform, args.mask)
    return dataset


def build_transform(is_train, args):

    t = []
    
    #if is_train:
        #t.append(transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0), interpolation=3))  # 3 is bicubic)
        #t.append(transforms.RandomHorizontalFlip())
   
    t.append(transforms.ToTensor())
   
    #t.append(transforms.Resize(56))
    
    #t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

from torch.utils.data import Dataset

class gas_loc_dataset(Dataset):
    def point2image(self,x,y,r):
        pose = torch.zeros((1,224,224))
    
        for i in range(pose.shape[1]):
            for j in range(pose.shape[2]):
                if pow(((224-j-x)**2+(i-y)**2),0.5)<r:
                    pose[:,i,j] = 1
        return pose
    def __init__(self, root, transform, mask_type):
        self.transform = transform  # using transform in torch!
        self.img_root = os.path.join(root, 'full_224')
        self.mask_root = os.path.join(root, mask_type)
        self.img_name_list = os.listdir( self.img_root)
        self.img_list = []
        self.mask_list = []
        self.pose = []
        self.img_name = []
     
        for dir in self.img_name_list:
           
            x,y,r = dir.split('-')
            x,y,r = float(x)/10,float(y)/10,float(r)/10
            pose = torch.tensor([x,y,r])
            
          
            for img in os.listdir(os.path.join(self.img_root,dir)):
                self.img_list.append(os.path.join( self.img_root, dir,img))
                self.mask_list.append(os.path.join( self.mask_root,dir, img))
                self.pose.append(pose)
                self.img_name.append(os.path.join(dir,img))
              
    

    
       
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
  
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx])
        img = self.transform(img)
        mask = self.transform(mask)
        
        #img = np.load(self.img_list[idx])
    
        
       
        # img = self.transform(img)
        # mask = self.add_mask(img)
      
        # img = img.float()
        # mask = mask.float()
       
        # from torchvision.transforms import ToPILImage
        # topil = ToPILImage()
        
        # image_pil = topil(mask)
        # label_pil = topil(img)
           
        # image_pil.save('image_pil.png')
        # label_pil.save('label_pil.png')
        # exit()
       
        return mask,img,self.pose[idx],self.img_name[idx]
    
    
class gas_mapping_dataset(Dataset):
    def __init__(self, root, transform, mask_type):
        self.transform = transform  # using transform in torch!
        self.img_root = os.path.join(root, 'full_224')
        self.mask_root = os.path.join(root, mask_type)
        self.img_name_list = os.listdir( self.img_root)
        self.img_list = []
        self.mask_list = []
        for img in self.img_name_list:
                self.img_list.append(os.path.join( self.img_root, img))
                self.mask_list.append(os.path.join( self.mask_root, img))
       
    
       
    def __len__(self):
        return len(self.img_list)
    def add_mask(self,img):
        org = torch.zeros((3,224,224))
        a = np.arange((7))
        x = a*2
        y = a*2
        for i in x:
            for j in y:         
                org[:,16*i:16*i+16,16*j:16*j+16] = img[:,16*i+7,16*j+7].unsqueeze(1).unsqueeze(1)
                #org[1:,16*i:16*i+16,16*j:16*j+16] = torch.mean(img[1:,16*i:16*i+16,16*j:16*j+16])
                #org[2:,16*i:16*i+16,16*j:16*j+16] = torch.mean(img[2:,16*i:16*i+16,16*j:16*j+16])
        return org
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx])
        #img = self.transform(img)
        mask = self.transform(mask)
        img = transforms.ToTensor()(img)
        #img = transforms.ToTensor()(img)
        #img = np.load(self.img_list[idx])
    
        
       
        # img = self.transform(img)
        # mask = self.add_mask(img)
      
        # img = img.float()
        # mask = mask.float()
       
        # from torchvision.transforms import ToPILImage
        # topil = ToPILImage()
        
        # image_pil = topil(mask)
        # label_pil = topil(img)
           
        # image_pil.save('image_pil.png')
        # label_pil.save('label_pil.png')
        # exit()
       
        return mask,img,self.img_name_list[idx]