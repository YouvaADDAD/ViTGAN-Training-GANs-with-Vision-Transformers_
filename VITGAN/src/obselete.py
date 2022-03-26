import torch 
from torch import nn 
from utils import Sine
import math
import torch.nn.functional as F

class ModulatedLinear(nn.Module):
    def __init__(self, in_features, out_features, demodulation=True):
        super(ModulatedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))        
        self.demodulation = demodulation


    def forward(self, Efou, y):
        #Efou ->(BxL,PxP,E)
        #y -> (B, L, E)
        #groups in conv1d apply in each num_peatches
        #Weight ->(out_features, in_features)
        #y -> (batch_size, 1, self.in_features)
        batch_size = Efou.shape[0]
        y = y.view(batch_size, 1, self.in_features) #(BxL,1,in_features)
        weight = self.scale * self.weight * y

        if self.demodulation:
            dconf = torch.rsqrt(weight.pow(2).sum(dim = 2) + 1e-8)
            weight = weight * dconf.view(batch_size, self.out_features, 1) # batch, out_features, in_features

        weight = weight.view(batch_size * self.out_features, self.in_features, 1)

        size_patch = Efou.size(1)
        Efou = Efou.reshape(1, batch_size * self.in_features, size_patch)
        x = F.conv1d(Efou, weight, groups=batch_size)
        x = x.view(batch_size, size_patch, self.out_features)
        return x

class Patch_Embedded(nn.Module):
    def __init__(self,image_shape, in_channel=3, patch_size=16, emb_dim=512):
        super(Patch_Embedded,self).__init__()
        #Pour faire du overlapping 
        H,W=image_shape
        #(H_in)/patch_size x (W_in)/patch_size
        L = ((H*W)//(patch_size**2)) 
        self.embedding = nn.Conv2d(in_channel, emb_dim, kernel_size=2*patch_size,stride=patch_size,padding=patch_size//2)
        #a learnable classification embedding xclass
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim)) 
        self.pos_emb = nn.Parameter(torch.randn(L+1, emb_dim))

    def forward(self, x):
        # x: shape BxCxHxW
        #return: B x L x emb_dim with L=H*W/P^2
        x=self.embedding(x)
        #TO DO add Pos Embedding + classification token
        x=x.flatten(start_dim=2).transpose(1, 2)
        cls_token=self.cls_token.repeat(x.shape[0],1,1)
        x = torch.cat((cls_token, x), dim=1)
        activation=Sine()
        return x + activation(self.pos_emb)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=8, discriminator=False, bias_qkv=False, bias_out_att=True, drop_attention=0., drop_out=0.):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.discriminator = discriminator
        assert self.head_dim * self.num_heads == self.emb_dim, "emb_dim must be divisible by num_heads"
        
        #Weight Tie if discriminator
        #Q = XW_{q} , K = XW_{k} , and V = XW_{v}
        if discriminator:
            #Spectral Normalization arXiv:1802.05957, As we mentioned above, the spectral norm σ(W) that 
            #we use to regularize each layer of the discriminator is the largest singular value of W
            #2.2 FAST APPROXIMATION OF THE SPECTRAL NORM σ(W) In the article
            self.in_proj_qkv=nn.Linear(emb_dim,2*emb_dim,bias=bias_qkv)
        else:
            self.in_proj_qkv=nn.Linear(emb_dim,3*emb_dim,bias=bias_qkv)
        
        self.drop_attention = nn.Dropout(drop_attention) if drop_attention > 0.0 else nn.Identity()
        self.drop_out = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()
        self.out_attention=nn.Linear(emb_dim,emb_dim,bias=bias_out_att)
    
    def forward(self, x):
        #x: B,N,E
        #return: B,N,E
        B, N, E = x.shape
        
        if self.discriminator:
            q_k,v=self.in_proj_qkv(x).chunk(2, dim=-1)
            
            #Same weight for Q and K
            Q = q_k.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            K = q_k.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            V = v.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            attn_weights = (-torch.cdist(Q, K, p=2) * self.scale).softmax(dim=-1) #Min distance, great probability
        else:
            #If not discriminator
            q,k,v = self.in_proj_qkv(x).chunk(3, dim=-1)#B,N,E for All
            
            Q = q.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            K = k.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            V = v.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            attn_weights = (torch.matmul(Q, K.transpose(-1, -2)) * self.scale).softmax(dim=-1)
        
        attn_weights = self.drop_attention(attn_weights)
        attn_out = (torch.matmul(attn_weights,V)).transpose(1, 2).reshape(B, N, E)
        attn_out = self.out_attention(attn_out)
        attn_out = self.drop_out(attn_out)
        
        return attn_out

class ModulatedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, demodulation=True, bias_init = 0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(1, out_features, in_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.ones(out_features) * bias_init)
        else:
            self.register_parameter('bias', None)
            
        self.demodulation = demodulation

    def forward(self, Efou, y):
        #Efou ->(BxL,PxP,E)
        #y -> (B, L, E)
        #groups in conv1d apply in each num_peatches
        #Weight ->(1, out_features, in_features, 1)
        #y -> (batch_size, 1, self.in_features, 1)

        batch_size = Efou.shape[0]
        y = y.view(batch_size, 1, self.in_features, 1)
        weight = self.scale * self.weight * y

        if self.demodulation:
            dnorm = torch.rsqrt(weight.pow(2).sum([2]) + 1e-8)
            weight = weight * dnorm.view(batch_size, self.out_features, 1, 1)

        weight = weight.view(batch_size * self.out_features, self.in_features, 1)

        img_size = Efou.size(1)
        Efou = Efou.reshape(1, batch_size * self.in_features, img_size)
        x = F.conv1d(Efou, weight, groups=batch_size, bias = self.bias)
        x = x.view(batch_size, img_size, self.out_features)
        return x

class ModulatedLinear2(nn.Module):
    def __init__(self, in_features, out_features, bias=False, demodulation=True, bias_init = 0.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(1, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.ones(out_features) * bias_init)
        else:
            self.register_parameter('bias', None)
            
        self.demodulation = demodulation

    def forward(self, Efou, y):
        #Efou ->(BxL,PxP,E)
        #y -> (B, L, E)
        #groups in conv1d apply in each num_peatches
        #Weight ->(1, out_features, in_features, 1)
        #y -> (batch_size, 1, self.in_features, 1)
        batch_size = Efou.shape[0]
        y = y.view(batch_size, 1, self.in_features)
        weight = self.scale * self.weight * y

        if self.demodulation:
            dconf = torch.rsqrt(weight.pow(2).sum(dim = 2) + 1e-8)
            weight = weight * dconf.view(batch_size, self.out_features, 1) # batch, out_features, in_features

        weight = weight.view(batch_size * self.out_features, self.in_features, 1)

        size_patch = Efou.size(1)
        Efou = Efou.reshape(1, batch_size * self.in_features, size_patch)
        x = F.conv1d(Efou, weight, groups=batch_size, bias = self.bias)
        x = x.view(batch_size, size_patch, self.out_features)
        return x

class ModulatedLinear3(nn.Module):
    def __init__(self, in_features, out_features, bias=False, demodulation=True, bias_init = 0.):
        super(ModulatedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.ones(out_features) * bias_init)
        else:
            self.register_parameter('bias', None)
            
        self.demodulation = demodulation

    def forward(self, Efou, y):
        #Efou ->(BxL,PxP,E)
        #y -> (B, L, E)
        #groups in conv1d apply in each num_peatches
        #Weight ->(out_features, in_features)
        #y -> (batch_size, 1, self.in_features)
        batch_size = Efou.shape[0]
        y = y.view(batch_size, 1, self.in_features) #(BxL,1,in_features)
        weight = self.scale * self.weight * y

        if self.demodulation:
            dconf = torch.rsqrt(weight.pow(2).sum(dim = 2) + 1e-8)
            weight = weight * dconf.view(batch_size, self.out_features, 1) # batch, out_features, in_features

        weight = weight.view(batch_size * self.out_features, self.in_features, 1)

        size_patch = Efou.size(1)
        Efou = Efou.reshape(1, batch_size * self.in_features, size_patch)
        x = F.conv1d(Efou, weight, groups=batch_size, bias = self.bias)
        x = x.view(batch_size, size_patch, self.out_features)
        return x
