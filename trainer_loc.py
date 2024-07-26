
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.nn.modules.loss import L1Loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.loss import EUDistLoss,WaveL1Loss

import timm
import timm.optim
import timm.scheduler

def get_mask(x):
    #mask = torch.ones((x.shape[0],1,x.shape[2],x.shape[3])).to(device)#1 is visible
    img = x*10000
    img = torch.round(img)
    mask = torch.where(img==4980,0.0,1.0)[:,0,:,:].unsqueeze(1)

  
    return mask

def image2point(pose_img):
    pose = torch.zeros((pose_img.shape[0],2))
    for i in range(pose_img.shape[0]):
        value = torch.max(pose_img[i])
       
        idx1,idx2 = torch.where((pose_img[i][0]-value)>=0)
        length = idx1.shape[0]
        y = idx1[int(length//2)]
        x = 224 - idx2[int(length//2)]
        pose[i] = torch.tensor([x,y])
    return pose

    
def add_mask(img_,mode,mask_ratio):
    img = img_.clone()
    mask_num = int(14*14*mask_ratio)
    #mask = torch.ones((img_.shape[0],1,img_.shape[2],img_.shape[3])).to('cuda:0')#1 is visible
    

    if mode=='random':
        a = list(np.arange((14*14)))
        mask_pos = random.sample(a,mask_num)
        for i in mask_pos:
            x , y = i//14*16 , i%14*16
            #print(x,y)
            img[:,:,x:x+16,y:y+16]= 0.498
        mask = torch.where(img==0.498,0.0,1.0)
        return img,mask

    if mode=='grid':
        org = np.ones((224,224,3))*0.498
        a = np.arange((7))
        x = a*2
        y = a*2
        for i in x:
            for j in y:
                org[:,16*i:16*i+16,16*j:16*j+16,:] = img[:,16*i:16*i+16,16*j:16*j+16,:]
        mask = torch.where(img==0.498,0,1)
        return org
    
def train_gas_loc(args, device, model, snapshot_path):
    #from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from datasets.dataset_gas import build_dataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    #db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                           transform=transforms.Compose(
    #                               [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_train = build_dataset(True, args)
    db_val =  build_dataset(False, args)
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                             #worker_init_fn=worker_init_fn)
    
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    wavel1_loss = WaveL1Loss(level=2)
    l1_loss =  L1Loss()
    mse_loss = EUDistLoss()


    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95))

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1

    warmup_it = args.warmup*len(trainloader)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer,t_initial=max_iterations,lr_min=1e-5,warmup_t=warmup_it,warmup_lr_init=1e-4)


    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
     
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, (image_batch,label_batch,pose_label,img_name) in enumerate(trainloader):
          
            #grid_train: no need to add mask
            image_batch,label_batch,pose_label = image_batch.to(device), label_batch.to(device), pose_label.to(device)

            mask = get_mask(image_batch)
          
            # from torchvision.transforms import ToPILImage
            # topil = ToPILImage()
            # mask_pil = topil(pose_img[2])
            # image_pil = topil(image_batch[2])
            # label_pil = topil(label_batch[2])
            # mask_pil.save('mask_inv.png')
            # image_pil.save('image_pil.png')
            # label_pil.save('label_pil.png')
            # exit()
      
            outputs, pose_prd = model(image_batch,mask)
        
        
            loss1  = l1_loss(outputs, label_batch)
            loss2 = wavel1_loss(outputs, label_batch)
            
            
            loss_loc = l1_loss(pose_prd,pose_label[:,0:2])
            loss = loss_loc
            loss = loss1+loss2+loss_loc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          

            scheduler.step(iter_num)
    
            lr_ = optimizer.param_groups[0]['lr']

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loc_loss', loss_loc, iter_num)
          
       

            #logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

        if  (epoch_num + 1) % args.save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        total_loss = 0
        total_loc = 0
        if (epoch_num+1) % args.eval_interval ==0:
            model.eval()
            for i_batch, (image_batch,label_batch,pose_label,img_name) in enumerate(valloader):
                with torch.no_grad():
                    image_batch, label_batch,pose_label = image_batch.to(device), label_batch.to(device),pose_label.to(device)
                    
                    mask = get_mask(image_batch)
                    outputs,pose_prd = model(image_batch,mask)
               
                
                    loss = l1_loss(outputs, label_batch)
               

                    loss_loc = mse_loss(pose_label[:,0:2], pose_prd)
                    total_loss = total_loss + loss.item()
                    total_loc = total_loc + loss_loc.item()

            total_loss = total_loss/len(valloader)
            total_loc = total_loc/len(valloader)
            writer.add_scalar('val/val_loss', total_loss, epoch_num)
            writer.add_scalar('val/loss_loc', total_loc, epoch_num)
            logging.info('epoch:'+str(epoch_num)+' val loss:'+str(total_loss)+'  loc_loss:'+str(total_loc))
       
       
  
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"