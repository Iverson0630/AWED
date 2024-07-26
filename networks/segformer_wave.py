from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from utils.attention import EfficientSelfAttention, MaskAwareSelfAttention, EfficientMaskedAttention
from wavemix import Level1Waveblock
# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class PreNormWithMask(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, mask):
        return self.fn(self.norm(x),mask)
    



class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        #stage_kernel_stride_pad = ((4, 4, 0), (2, 2, 0), (2, 2, 0), (2, 2, 0))
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))
        
        self.stages = nn.ModuleList([])
        
        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)
            fused_conv = nn.Conv2d(dim_out*2 , dim_out, 1)
            layers = nn.ModuleList([])
            wave_block = nn.ModuleList([])
            ##add wave block
            depth= 8
            mult = 2
          
            dropout = 0.5
            for _ in range(depth):
                wave_block.append(Level1Waveblock(mult = mult, ff_channel = dim_out, final_dim = dim_out, dropout = dropout))


            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                wave_block,
                layers,
                fused_conv
            ]))

    def forward(
        self,
        x,
        mask,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for idx,(get_overlap_patches, overlap_embed, wave_block, layers,fused_conv) in enumerate(self.stages):
       
            x = get_overlap_patches(x)
            # mask = get_overlap_patches(mask)
            # mask = mask.mean(dim=1).unsqueeze(1)
            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)
            # mask = rearrange(mask, 'b c (h w) -> b c h w', h = h // ratio)
           
            wave = overlap_embed(x)
            trans = overlap_embed(x)
          
            for (attn, ff) in layers:
                trans = attn(trans) + trans
                trans = ff(trans) + trans
            x = trans
            if idx<3:
                for attn in wave_block:
                    wave = attn(wave) + wave
                x = torch.cat([trans,wave],dim=1)
              
                x = fused_conv(x)
         
            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret
    
class Decoder(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        placeholders = (1,)
        heads,ff_expansion,reduction_ratio = placeholders+heads[0:-1],placeholders+ff_expansion[0:-1],placeholders+reduction_ratio[0:-1]
        self.stages = nn.ModuleList([])
        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            fused_conv = nn.Conv2d(dim_in*2 , dim_in, 1)
            layers = nn.ModuleList([])
            wave_block = nn.ModuleList([])

            depth= 8
            mult = 2
          
            dropout = 0.5
            for _ in range(depth):
                wave_block.append(Level1Waveblock(mult = mult, ff_channel = dim_in, final_dim = dim_in, dropout = dropout))

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_in, EfficientSelfAttention(dim = dim_in, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_in, MixFeedForward(dim = dim_in, expansion_factor = ff_expansion)),
                ]))
            self.stages.append(nn.ModuleList([
                #nn.ConvTranspose2d(dim_out,dim_out,3, stride=2, padding=1,output_padding=1),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim_in+dim_out, dim_in, 1),
                layers,
                wave_block,
                fused_conv 
            ]))
        

    def forward(self,output):
        x = output[-1]

        for idx,(unsample,conv,layers,wave_block,fused_conv) in enumerate(reversed(self.stages)):
            if idx>2:
                return x
            
            x = unsample(x)
           
            x = torch.concat((x,output[-2-idx]),dim=1)
            x = conv(x)
            wave = x
            trans = x
          
            for (attn, ff) in layers:
                trans = attn(trans) + trans
                trans = ff(trans) + trans
            x = trans
            if idx>0:
                for attn in wave_block:
                    wave = attn(wave) + wave
                x = torch.cat([trans,wave],dim=1)
              
                x = fused_conv(x)
            
     
        
     
     


class SegformerWave(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 4,
        decoder_dim = 256,
        num_classes = 3
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        # self.to_fused = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(dim, decoder_dim, 1),
        #     nn.Upsample(scale_factor = 2 ** i),
        #     nn.Conv2d(decoder_dim, decoder_dim, 1),
        #     nn.Upsample(scale_factor = 4)
        # ) for i, dim in enumerate(dims)])

      
        self.decoder = Decoder(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )
        
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(32, num_classes, 1),
            nn.Upsample(scale_factor = 4),
        )
    def encode(self,x,mask):
        x = torch.cat([x,mask],dim=1)
        return  self.mit(x, mask, return_layer_outputs = True)
    
    def decode(self,layer_outputs):
        return self.to_segmentation(self.decoder(layer_outputs))
    
    def forward(self, x, mask):
    
        layer_outputs = self.encode(x,mask)
        #fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        #fused = torch.cat(fused, dim = 1)
        return self.decode(layer_outputs)
