import os
from PIL import Image
import numpy as np
import random

root = 'data1k/train/'
gt_root = 'data1k/train/full_224/'
mode = '7_7' #random,grid , sway ,grid_mean
outdir = os.path.join(root,mode)
mask_ratio = 0.75
def add_mask(img,mode,mask_ratio):
    mask_num = int(14*14*mask_ratio)

    if mode=='7_7':
        org = np.ones((7,7,3))*127
        a = np.arange((7))
   
        for i in a:
            for j in a:
                org[i,j,:] = np.mean(np.mean(img[16*i*2:16*i*2+16,16*j*2:16*j*2+16,:],axis=0),axis=0)
        return org
    

    if mode=='random':
        a = list(np.arange((14*14)))
        mask_pos = random.sample(a,mask_num)
        for i in mask_pos:
            x , y = i//14*16 , i%14*16
            #print(x,y)
            img[x:x+16,y:y+16,:]= 127
        return img

    if mode=='grid':
        org = np.ones((224,224,3))*127
        a = np.arange((7))
        x = a*2
        y = a*2
        for i in x:
            for j in y:
                org[16*i:16*i+16,16*j:16*j+16,:] = img[16*i:16*i+16,16*j:16*j+16,:]
        return org
    
    if mode=='grid_mean':
        org = np.ones((224,224,3))*127
        a = np.arange((7))
        x = a*2
        y = a*2
        for i in x:
            for j in y:
              
                org[16*i:16*i+16,16*j:16*j+16,:] = np.mean(np.mean(img[16*i:16*i+16,16*j:16*j+16,:],axis=0),axis=0)
        return org
    
    if mode=='sway(0.75)_mean':
        org = np.ones((224,224,3))*127
        a = np.arange((16))
        for i in a:
            for j in a:
                if i%4==0:
                    org[14*i:14*i+14,14*j:14*j+14,:] = np.mean(np.mean( img[14*i:14*i+14,14*j:14*j+14,:],axis=0),axis=0)  
                if (i%8<4)and(j==15):    
                    org[14*i:14*i+14,14*j:14*j+14,:] = np.mean(np.mean( img[14*i:14*i+14,14*j:14*j+14,:],axis=0),axis=0)    
                if (i%8>3)and(j==0):
                    org[14*i:14*i+14,14*j:14*j+14,:] = np.mean(np.mean( img[14*i:14*i+14,14*j:14*j+14,:],axis=0),axis=0)    
        return org
    
    if mode=='sway(0.75)':
        org = np.ones((224,224,3))*127
        a = np.arange((16))
        for i in a:
            for j in a:
                if i%4==0:
                    org[14*i:14*i+14,14*j:14*j+14,:] = img[14*i:14*i+14,14*j:14*j+14,:]
                if (i%8<4)and(j==15):    
                    org[14*i:14*i+14,14*j:14*j+14,:] = img[14*i:14*i+14,14*j:14*j+14,:]
                if (i%8>3)and(j==0):
                    org[14*i:14*i+14,14*j:14*j+14,:] = img[14*i:14*i+14,14*j:14*j+14,:]  
        return org


for name in os.listdir(gt_root):
    img_path = os.path.join(gt_root,name)
    img = Image.open(img_path)
  
    img = img.convert("RGB") 
    
    out = np.array(img)
  
    out = add_mask(out,mode,mask_ratio)
    out = Image.fromarray(np.uint8(out))
    out.save(os.path.join(outdir,name))