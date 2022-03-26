from Modules import *
from FourierModules import *
from utils import *
from DiffAugment import *
import torch
import torch.nn.functional as F
from torch import nn  



class DisBlock(nn.Module):
    def __init__(self, emb_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, bias_out_att=True, drop_out=0., drop_attention=0., drop_path=0., activation=nn.GELU, norm_layer=nn.LayerNorm):
        super(DisBlock,self).__init__()
        self.norm1 = norm_layer(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads=num_heads, discriminator=True, bias_qkv=qkv_bias, bias_out_att=bias_out_att, drop_attention=drop_attention, drop_out=drop_attention)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.dropout = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(emb_dim)
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = MLP(emb_dim, hidden_dim=mlp_hidden_dim, activation=activation, dropout=drop_out, discriminator=True)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class Discriminator(nn.Module):
    def __init__(self, emb_dim=384, image_size=32, patch_size=4, in_channels=3, n_blocks=4,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, bias_out_att=True, drop_out=0., 
                 drop_attention=0., drop_path=0., activation=nn.GELU, norm_layer=nn.LayerNorm):
        super(Discriminator, self).__init__()

        self.emb_dim = emb_dim  
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_channels=in_channels, emb_dim=emb_dim)
        self.num_patches = self.patch_embedding.num_patches
        self.pos_emb = PatchPositionalEmbedding(emb_dim, self.num_patches, cls_token = True) # (N+1, E)
        self.norm = norm_layer(emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        
        self.blocks = nn.ModuleList([DisBlock(emb_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, bias_out_att=bias_out_att, drop_out=drop_out, 
                                              drop_attention=drop_attention, drop_path=drop_path, activation=activation, norm_layer=norm_layer)for _ in range(n_blocks)])
        
        #Real or True
        self.final_layer = nn.Linear(self.emb_dim, 1) 

        
    def forward(self, x):
        x , _ = self.patch_embedding(x) # Having patch So Size (B,N,E)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1) 
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_emb() # (B, N+1, E) Knows automaticly to broadcast

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)[:, 0]
        return self.final_layer(x)




