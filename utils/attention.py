from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import torch

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
       
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
      
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

       
        attn = sim.softmax(dim = -1)
  
        out = einsum('b i j, b j d -> b i d', attn, v)
      
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        
        return self.to_out(out)
    
class MaskAwareSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias = False)
   
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x, mask):
        
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
      
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))
        mask = rearrange(mask, 'b c h w  -> b (h w) c')
        mask = mask.repeat(heads,1,1) #multi-heads
      
        sim = einsum('b i d, b j d -> b i j', q, k)* mask * self.scale
        attn = sim.softmax(dim = -1)
    
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class EfficientMaskedAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.eps = 1e-8
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
       
        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_mask = nn.Unfold(reduction_ratio, stride = reduction_ratio)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)
   

    def forward(self, x, mask):
        
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
      
        mask_down = self.to_mask(mask)
        mask = rearrange(mask, 'b c h w  -> b (h w) c')
        mask_down = mask_down.mean(dim=1).unsqueeze(1)
        mask = (einsum('b i d, b j d -> b i d', mask, mask_down))
       
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))
       
        mask = mask.repeat(heads,1,1) #multi-heads
        
        mask = torch.log(mask + self.eps)  
        #mask = (mask-0.5)*5
  
        sim = (einsum('b i d, b j d -> b i j', q, k)+mask)* self.scale
        attn = sim.softmax(dim = -1)
      
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)