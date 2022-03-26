from Modules import *
from FourierModules import *
from utils import *
from DiffAugment import *
import torch
import torch.nn.functional as F
from torch import nn  
import math
import numpy as np


class GenBlock(nn.Module):
    def __init__(self, emb_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, bias_out_att=True, drop_out=0., drop_attention=0., drop_path=0., activation=nn.GELU, norm_layer=SLN):
        super(GenBlock,self).__init__()
        self.norm1 = norm_layer(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads=num_heads, discriminator=False, bias_qkv=qkv_bias, bias_out_att=bias_out_att, drop_attention=drop_attention, drop_out=drop_attention)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.dropout = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(emb_dim)
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = MLP(emb_dim, hidden_dim=mlp_hidden_dim, activation=activation, dropout=drop_out, discriminator=False)

    def forward(self, x, w):
        x = x + self.dropout(self.attn(self.norm1(x, w)))
        x = x + self.dropout(self.mlp(self.norm2(x, w)))
        return x

class Generator(nn.Module):
    def __init__(self, emb_dim=384, latent_dim = 100,image_size=32, patch_size=4, out_features=3, n_blocks=4,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, bias_out_att=True, drop_out=0., demodulation=True,
                 drop_attention=0., drop_path=0., activation=nn.GELU, norm_layer=SLN):
        super(Generator, self).__init__()
        self.num_patches = (image_size//patch_size)**2
        self.image_size  =image_size
        self.patch_size  = patch_size
        self.emb_dim = emb_dim
        self.out_features = out_features
        self.pos_emb = PatchPositionalEmbedding(emb_dim, self.num_patches) # (N, E)
        self.cord_emb = CoordinatesPositionalEmbedding(emb_dim, patch_size)
        self.norm = norm_layer(emb_dim)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.mapping_network = MappingNetwork(latent_dim, emb_dim, n_layers=4) 
        self.blocks = nn.ModuleList([GenBlock(emb_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, bias_out_att=bias_out_att, drop_out=drop_out, 
                                              drop_attention=drop_attention, drop_path=drop_path, activation=activation, norm_layer=norm_layer) for _ in range(n_blocks)])
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.f_theta = SIRENModulated(emb_dim, out_features=out_features, demodulation=demodulation, n_layers=2)
    
    def forward(self, z):
        w = self.mapping_network(z)
        pos_emb = self.pos_emb().repeat(z.shape[0], 1, 1)
        h = pos_emb
        for block in self.blocks:
            h = block(h, w) #B,L,E
        
        y = self.norm(h, w)
        batch_num = y.shape[0] * y.shape[1]
        Efou = self.cord_emb().repeat(batch_num,1,1) # (BxL, PxP, E)
        rgb_s = self.f_theta(Efou, y) #(BxL, PxP, C) 
        rgb_s = rgb_s.view(-1, self.image_size//self.patch_size, self.image_size//self.patch_size, self.patch_size, self.patch_size, self.out_features)
        rgb_s = rgb_s.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.image_size, self.image_size, self.out_features) #B,H,W,C

        return rgb_s.permute(0,3,1,2) #B,C,H,W
        

        

         
        

