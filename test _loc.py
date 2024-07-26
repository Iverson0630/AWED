import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss import EUDistLoss
from torch.nn.modules.loss import MSELoss,L1Loss
from torchvision import transforms
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_gas import build_dataset
from networks.segformer_pytorch import Segformer
from networks.wavepaint import WavePaint
from networks.segformer_wave import SegformerWave
from networks.unet import UNET
from networks.waveunet import WUNET
from PIL import  ImageDraw
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,  default='Gas', help='experiment_name')
parser.add_argument('--decoder', type=str,
                    default='transblock', help='transblock or mlp')
parser.add_argument('--model', type=str,
                    default='segformer', help='segformer or transunet')
parser.add_argument('--model_path', type=str,  default='output/data1k/grid_mean_train/wunet_32_loctransblock/_50k_epo400_bs32_lr0.001_224/epoch_399.pth')  
parser.add_argument('--mask', type=str,
                    default='grid_mean', help='mask mode:random or grid or sway(0.75)')             
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--save_path', type=str, default='../../data/data1k/gas_loc/result/', help='save_path')
parser.add_argument('--localization', type=bool,
                    default=True, help='whether to train gas localization network. if false, only train mapping network')

parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
args = parser.parse_args()

device = 'cuda:0'

def get_mask(x):
    #mask = torch.ones((x.shape[0],1,x.shape[2],x.shape[3])).to(device)#1 is visible
    img = x*10000
    img = torch.round(img)
    mask = torch.where(img==4980,0.0,1.0)[:,0,:,:].unsqueeze(1)
  
    return mask

def inference(args, model, test_save_path):
    db_test =  build_dataset(False, args)
    print("The length of test set is: {}".format(len(db_test)))
    toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    total_loss = 0
    loc_loss = 0
    l1_loss = L1Loss()
    mse_loss = EUDistLoss()

    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    count = 0.0
    draw_size = 4
    for i_batch, (image_batch,label_batch,pose_label,img_name) in tqdm(enumerate(testloader)):
        image_batch, label_batch,pose_label = image_batch.to(device), label_batch.to(device),pose_label.to(device)
        mask = get_mask(image_batch)
        
        outputs, pose_pre = model(image_batch,mask)
        #pred = outputs*(1-mask)+image_batch*mask
        for i in range(len(outputs)):
            img_tensor = outputs[i].cpu()
            x_pred, y_pred = 224 - pose_pre[i][0].item()*224,pose_pre[i][1].item()*224
            x_label, y_label = 224-pose_label[i][0]*224, pose_label[i][1]*224 

            img_path = os.path.join(test_save_path, img_name[i].replace('/','_'))
            img_png = toPIL(img_tensor)  
            draw = ImageDraw.Draw(img_png)
            draw.polygon([(x_label-draw_size,y_label-draw_size),(x_label-draw_size,y_label+draw_size),
                          (x_label+draw_size,y_label+draw_size),(x_label+draw_size,y_label-draw_size)],fill='#c1cdc1')
            draw.polygon([(x_pred,y_pred-draw_size),(x_pred-draw_size,y_pred+draw_size),(x_pred+draw_size,y_pred+draw_size)],fill='#FF00FF')
            img_png.save(img_path)
           
        loss_loc = mse_loss(pose_label[:,0:2], pose_pre)
      
        loss = l1_loss(outputs, label_batch)
        total_loss = total_loss+ loss.item()
        loc_loss = loc_loss + loss_loc.item()
    total_loss = total_loss/len(testloader)
    print(' test loss:'+str(total_loss))
    return "Testing Finished!"


if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Gas': {
            'root_path': '../../data/data1k/gas_loc',
            'num_classes': 3,
        },

    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
  
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    # if args.vit_name.find('R50') !=-1:
    #     config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    # if args.model =='transunet':
    #     net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    #     net.load_from(weights=np.load(config_vit.pretrained_path))
    # elif args.model =='segformer':
    #     net = Segformer(
    #         dims = (32, 64, 160, 256),      # dimensions of each stage
    #         heads = (1, 2, 5, 8),           # heads of each stage
    #         ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    #         reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    #         num_layers = 2,                 # num layers of each stage
    #         decoder_dim = 256,              # decoder dimension
    #         num_classes = 3 ,                # number of segmentation classes
    #         decoder = args.decoder
    #          )
    # #         dims = (64,128,320,512),      # dimensions of each stage
    # #         heads = (1, 2, 5, 8),           # heads of each stage
    # #         ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
    # #         reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
    # #         num_layers = 3,                 # num layers of each stage
    # #         decoder_dim = 512,              # decoder dimension
    # #         num_classes = 3,                 # number of segmentation classes
    # #         decoder = args.decoder
    # #         )   
    #net = SegformerWave()

    net = WUNET(in_channels=4, out_channels=3)
    net = net.to(device)
    net.load_state_dict(torch.load(args.model_path))
    print('load from:'+args.model_path)
    save_dir = os.path.join(args.save_path,args.mask)
    os.makedirs(save_dir, exist_ok=True)

    inference(args, net, save_dir)


