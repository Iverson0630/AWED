from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.transforms import ToTensor,ToPILImage
import pywt

# -*- coding: utf-8 -*-
from torch.autograd import Variable
from pytorch_wavelets import DWTForward, DWTInverse





dct = DWTForward(J=1, mode='zero', wave='haar').cuda()
idct = DWTInverse(mode='zero', wave='haar').cuda()


class WRB(nn.Module):# wavelet reconstruction block
    def __init__(self, c):
        super().__init__()
        self.conv= nn.Conv2d(c,c//4,1)
        self.mlp= nn.Sequential(
            nn.Conv2d(c,c,1),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(c,c*4,1),
        )
        
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c, c//2 , kernel_size=2, stride=2),
            nn.BatchNorm2d(c//2),
        )

    def wt(self,vimg):
        l1, yh = dct(vimg)
       
        y = yh[0].reshape(vimg.shape[0],-1,int(vimg.shape[2]/2),int(vimg.shape[3]/2))   
        return torch.cat([l1,y],dim=1)


    def iwt(self,vres):
        chanel = int(vres.shape[1]/4)
        l1 = vres[:,0:chanel,:]
        yh = vres[:,chanel:,:].reshape(vres.shape[0],chanel,3,vres.shape[2],vres.shape[3])
        yh = list([yh])
        res = idct((l1,yh))
    
        return res

    def forward(self, x):

        x = self.conv(x)
       
        x = self.wt(x)
        x = self.mlp(x)
        x = self.iwt(x)

    

        x = self.up(x)

        return x

class WRM(nn.Module):# wavelet reconstruction module
    def __init__(self, c , num_lay):
        super().__init__()
        self.layers =  nn.ModuleList([])
        
        for _ in range(num_lay):
            self.layers.append(WRB(c))
     
    def forward(self, x):
        for lay in self.layers:
            x = x + lay(x) 
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dilation, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DoubleConv_Down(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dilation, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=16, padding=1,stride=16, bias=False,dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_One(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, 1, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #input is CHW
        return self.conv(x1)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, 1, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2 , kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
       
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class WUNET(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, device: str="cuda"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.device = device
        
   
        factor = 2 if bilinear else 1
        base_c = 64
        self.inc1 = (DoubleConv(in_channels, base_c,dilation=1))
        #self.inc2 = (DoubleConv(base_c, base_c*4,dilation=1))
        self.down1 = (Down(base_c, base_c*2,dilation=1))
    
        self.down2 = (Down(base_c*2, base_c*4,dilation=1))
       
        self.down3 = (Down(base_c*4, base_c*8,dilation=1))
       
        self.down4 = (Down(base_c*8, base_c*16 // factor,dilation=1))
        
        #self.down = DoubleConv_Down(base_c*4, base_c*16,dilation=1)

        self.iwt1 = WRB(base_c*16)
        self.iwt2 = WRB(base_c*8)
        self.iwt3 = WRB(base_c*4)
        self.iwt4 = WRB(base_c*2)

        self.up1 = (Up(base_c*16, base_c*8 // factor, bilinear))
  
        self.up2 = (Up(base_c*8, base_c*4 // factor, bilinear))
      
        self.up3 = (Up(base_c*4, base_c*2 // factor, bilinear))

        self.up4 = (Up(base_c*2, base_c, bilinear))
       
        self.outc = (OutConv(base_c, out_channels))
        #self.outc_loc = LocHead(base_c*2, 1, 112, 2)

    def forward(self, x, masks):
     
        x = torch.cat([x, masks], dim=1)
        x1 = self.inc1(x)
      
        #one step downsample

        # x1 = self.inc2(x1)
        # x = self.down(x1)
        # x = self.up1(x, x)+self.iwt1(x) # 6 -> 12
        # x = self.up2(x, x)+self.iwt2(x) # 12 -> 24
        # x = self.up3(x, x)+self.iwt3(x) # 24 -> 48

        # x = self.up4(x, x)+self.iwt4(x) # 48 -> 96





        # multi downsample
        x2 = self.down1(x1) # 96 -> 48
        x3 = self.down2(x2) # 48 -> 24
        x4 = self.down3(x3) # 24 -> 12
        x5 = self.down4(x4) # 12 -> 6
      
       


        x = self.up1(x5, x4)+self.iwt1(x5) # 6 -> 12
        x = self.up2(x, x3)+self.iwt2(x) # 12 -> 24
        x = self.up3(x, x2)+self.iwt3(x) # 24 -> 48
        #loc_out = self.outc_loc(x)
        x = self.up4(x, x1)+self.iwt4(x) # 48 -> 96
        
    
        raw_out = self.outc(x)
       
        
        # imputed_data = masks * X + (1 - masks) * raw_out

        return raw_out