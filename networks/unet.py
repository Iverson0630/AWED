import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, 1, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
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

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, device: str="cuda"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.device = device

        factor = 2 if bilinear else 1
        base_c = 32
        self.inc = (DoubleConv(in_channels, base_c,dilation=1))
        self.down1 = (Down(base_c, base_c*2,dilation=1))
    
        self.down2 = (Down(base_c*2, base_c*4,dilation=1))
       
        self.down3 = (Down(base_c*4, base_c*8,dilation=1))
       
        self.down4 = (Down(base_c*8, base_c*16 // factor,dilation=1))
        
      

        self.up1 = (Up(base_c*16, base_c*8 // factor, bilinear))

        self.up2 = (Up(base_c*8, base_c*4 // factor, bilinear))

        self.up3 = (Up(base_c*4, base_c*2 // factor, bilinear))

        self.up4 = (Up(base_c*2, base_c, bilinear))

        self.outc = (OutConv(base_c, out_channels))

    def forward(self, X, masks):
        x = torch.cat([X, masks], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1) # 96 -> 48
        x3 = self.down2(x2) # 48 -> 24
        x4 = self.down3(x3) # 24 -> 12
        x5 = self.down4(x4) # 12 -> 6
        x = self.up1(x5, x4) # 6 -> 12
        x = self.up2(x, x3) # 12 -> 24
        x = self.up3(x, x2) # 24 -> 48
        x = self.up4(x, x1) # 48 -> 96
        # x = self.up4(x2, x1) # 48 -> 96
        raw_out = self.outc(x)
        
        imputed_data = masks * X + (1 - masks) * raw_out

        return raw_out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)