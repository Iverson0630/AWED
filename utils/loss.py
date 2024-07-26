
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class FocalL1Loss(nn.Module):
    def __init__(self,gama):
        super(FocalL1Loss, self).__init__()
        self.gama = gama
    def forward(self, inputs, target):
    
        loss = F.l1_loss(inputs, target,reduce=False).mean(dim=1)
      
        loss_ =  F.normalize(1/loss,dim=0)
                                                                                                                                                          
        weight = -torch.log(loss_)*(1-loss_)**self.gama*2
   
        return loss.mean(), torch.mul(loss,weight).mean()
    
class EUDistLoss(nn.Module):
    def __init__(self):
        super(EUDistLoss, self).__init__()
   
    def forward(self, a, b):
        x1 = pow(a[:,0]-b[:,0],2)
        x2 = pow(a[:,1]-b[:,1],2)
   
        return pow((x1+x2),0.5).mean()

class WaveL1Loss(nn.Module):
    def __init__(self,level):
        super(WaveL1Loss, self).__init__()
        self.level = level
        self.dct = DWTForward(J=level, mode='zero', wave='db1').cuda()

    def wt(self,vimg):

        l1, yh = self.dct(vimg)

        
        if self.level==1:
            y = yh[0].reshape(vimg.shape[0],-1,int(vimg.shape[2]/2),int(vimg.shape[3]/2))   
            return torch.cat([l1,y],dim=1)
        elif self.level==2:
            y0 = yh[0]
            y1 = yh[1]
            l1 = l1.unsqueeze(2)
            return y0, torch.cat([l1,y1],dim=2)

    def forward(self, inputs, target):
        if self.level==1:
            target = self.wt(target)
            inputs = self.wt(inputs)
            loss = F.l1_loss(inputs, target)
        elif self.level==2:
            target1,target2 = self.wt(target)
      
            inputs1, inputs2 = self.wt(inputs)
            loss = F.l1_loss(inputs1, target1)+F.l1_loss(inputs2, target2)
        return loss
