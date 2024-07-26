import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import train_gas
from trainer_loc import train_gas_loc
from utils.init_para import weight_init
from networks.segformer_pytorch import Segformer
from networks.DRCT_arch import DRCT
from networks.wavepaint import WavePaint
from networks.segformer_wave import SegformerWave
from networks.unet import UNET
from networks.waveunet import WUNET
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='Gas', help='experiment_name')
parser.add_argument('--model_path', type=str,  default='output/data1k/grid_mean_train/wnet_64_wavel1mlp/_50k_epo400_bs16_lr0.001_224/epoch_399.pth')  
parser.add_argument('--model', type=str,
                    default='unet32_wavel1', help='segformer or transunet or unet wnet')
parser.add_argument('--decoder', type=str,
                    default='mlp', help='transblock or mlp')
parser.add_argument('--mask', type=str,
                    default='grid_mean', help='mask mode:random or grid or sway(0.75)')
parser.add_argument('--mask_ratio', type=float,
                    default=0.75, help='mask ratio')
parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--localization', type=bool,
                    default=False, help='whether to train gas localization network. if false, only train mapping network')
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--warmup', type=int,
                    default=40, help='warmup epochs')
parser.add_argument('--save_interval', type=int,
                    default=40, help='save_interval')
parser.add_argument('--eval_interval', type=int,
                    default=10, help='eval_interval')
device = 'cuda:0'
args = parser.parse_args()
def get_para(model):
    import thop
    x = torch.randn(1, 3, 224, 224).cuda()
    mask = torch.randn(1, 1, 224, 224).cuda()
    flops, params = thop.profile(model, inputs=(x,mask))
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("------|-----------|------")
    print("%s | %.7f | %.7f" % ("模型  ", params / (1000 ** 2), flops / (1000 ** 3)))

def get_fps(model):
    model.eval()
    import time
    x = torch.zeros((1,3, 7, 7)).cuda()
    mask = torch.randn(1, 1, 224, 224).cuda()
    t_all = []

    for i in range(100):
        t1 = time.time()
        y = model(x,mask)
        t2 = time.time()
        t_all.append(t2 - t1)

    print('average time:', np.mean(t_all) / 1)
    print('average fps:',1 / np.mean(t_all))

    print('fastest time:', min(t_all) / 1)
    print('fastest fps:',1 / min(t_all))

    print('slowest time:', max(t_all) / 1)
    print('slowest fps:',1 / max(t_all))


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True



    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Gas': {
            'root_path': '../../data/data1k',
            'num_classes': 3,
        },

    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.is_pretrain = True
    args.exp = args.model + args.decoder
    snapshot_path = "output/data1k/grid_mean_train/{}/".format(args.exp)
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    
    if args.model =='transunet':
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
        #net.load_from(weights=np.load(config_vit.pretrained_path))
    elif args.model =='segformer':
        net = Segformer(
            dims = (32, 64, 160, 256),      # dimensions of each stage
            heads = (1, 2, 5, 8),           # heads of each stage
            ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
            reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
            num_layers = 2,                 # num layers of each stage
            decoder_dim = 256,              # decoder dimension
            num_classes = 3,                 # number of segmentation classes
            decoder = args.decoder
        
            )
   
    net = UNET(in_channels=4, out_channels=3)
    net = net.to(device)

    # get_fps(net)
    # exit()
    # get_para(net)
    # exit()

  
    # net.load_state_dict(torch.load(args.model_path),strict=False)
    # print('load from:'+args.model_path)
    #net.apply(weight_init)
    # net = WavePaint()
    #print(net)
    # net = SegformerWave()
    # net = net.to(device)
    # for name,para in net.named_parameters():
    #     print(name)

   
    trainer = {'Gas':train_gas}
    trainer[dataset_name](args,device, net, snapshot_path)