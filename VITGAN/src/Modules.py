from torch import nn
import torch
from torch.nn.functional import normalize #v=\frac{v}{\max (\|v\|_{p}, \epsilon)}
import torch.nn.functional as F
from utils import *
                                                                                                                                                                                                             
class PatchEmbedding(nn.Module):
    def __init__(self, image_size = 32, patch_size = 4, in_channels = 3, emb_dim = 384):
        #Patch, Embedding, overlapping
        super(PatchEmbedding,self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.conv = nn.Conv2d(in_channels, emb_dim, kernel_size = 2 * patch_size, stride = patch_size, padding = patch_size//2)
        self.norm = nn.BatchNorm2d(emb_dim)

    def forward(self, x):
        # x: shape BxCxHxW
        #return: B x L x emb_dim with L=H*W/P^2
        _, _, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        H, W = H // self.patch_size, W // self.patch_size
        assert H * W == x.shape[1], 'May have the same dimension'
        return x, (H, W)

class PatchPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, sequence_length, cls_token = False):
        #Cls_token for discriminator if equal to true (N+1, E)
        #else Generator (N,E)
        super(PatchPositionalEmbedding, self).__init__()
        self.pos_emb = nn.Linear(1, emb_dim)
        self.sequence_length = sequence_length
        self.cls_token = cls_token
    
    def forward(self):
        if self.cls_token:
            x = torch.linspace(-1, 1, self.sequence_length + 1, requires_grad=False).unsqueeze(1).cuda()
        else:
            x =  torch.linspace(-1, 1, self.sequence_length , requires_grad=False).unsqueeze(1).cuda()
        return torch.sin(self.pos_emb(x))

class CoordinatesPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, patch_size, dim = 2):
        super(CoordinatesPositionalEmbedding, self).__init__()
        self.pos_emb = nn.Linear(dim, emb_dim)
        self.patch_size = patch_size
        self.dim = dim
    
    def forward(self):
        coordinates = tuple(self.dim * [torch.linspace(-1, 1, self.patch_size, requires_grad=False)])
        mgrid = torch.stack(torch.meshgrid(*coordinates),  dim=-1) #, indexing="xy"
        mgrid = mgrid.reshape(-1, self.dim).cuda()
        x = self.pos_emb(mgrid)
        return torch.sin(x)

#The MLP contains two layers with a GELU non-linearity. (arXiv:2010.11929)
class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim=None, out_dim=None, activation=nn.GELU, dropout=0.0, discriminator = False):
        super(MLP, self).__init__()    
        out_dim = out_dim or d_model
        hidden_dim = hidden_dim or d_model
        # why using GeLu see article arXiv:1606.08415 
        if discriminator:
            self.model = nn.Sequential(SpectralNorm(nn.Linear(d_model,hidden_dim)),
                                       activation(),
                                       nn.Dropout(p=dropout),
                                       SpectralNorm(nn.Linear(hidden_dim,out_dim)),
                                       nn.Dropout(p=dropout))
        else:
            self.model = nn.Sequential(EqualizedLinear(d_model,hidden_dim, bias_init=1),
                                       activation(),
                                       nn.Dropout(p=dropout),
                                       EqualizedLinear(hidden_dim,out_dim, bias_init=1),
                                       nn.Dropout(p=dropout))   
    def forward(self, x):
        return self.model(x)

#The Lipschitz Constant of Self-Attention, arXiv:2006.04710
#We'll have Two case discriminator and generator
#Stabilizing Discriminator= L2 Attention+Spectral norm
#Enforcing Lipschitzness of Transformer Discriminator:  arXiv:2006.04710
#MultiHead(Q,K,V)=Concat(head_{1},...,head_{h})WO
#head_{i}=Attention(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V})
#TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up : arXiv:2102.07074
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=8, discriminator=False, bias_qkv=False, bias_out_att=True, drop_attention=0., drop_out=0.):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        #self.h_w_size = h_w_size, Tuple of (H,W)
        self.scale = self.head_dim ** -0.5
        self.discriminator = discriminator
        assert self.head_dim * self.num_heads == self.emb_dim, "emb_dim must be divisible by num_heads"
        
        #Weight Tie if discriminator
        #Q = XW_{q} , K = XW_{k} , and V = XW_{v}
        self.drop_attention = nn.Dropout(drop_attention) 
        self.drop_out = nn.Dropout(drop_out)
        self.out_attention = nn.Linear(emb_dim,emb_dim,bias=bias_out_att)
        
        if discriminator:
            #Spectral Normalization arXiv:1802.05957, As we mentioned above, the spectral norm σ(W) that 
            #we use to regularize each layer of the discriminator is the largest singular value of W
            #2.2 FAST APPROXIMATION OF THE SPECTRAL NORM σ(W) In the article 

            #The Lipschitz Constant of Self-Attention
            #Weight Tying
            #arXiv:2006.04710
            self.proj_qk = SpectralNorm(nn.Conv2d(emb_dim , emb_dim, kernel_size = 3, stride = 1, padding = 1 , bias=bias_qkv))
            self.proj_v  = SpectralNorm(nn.Conv2d(emb_dim , emb_dim, kernel_size = 3, stride = 1, padding = 1 , bias=bias_qkv))
            #self.proj_qk = SpectralNorm(nn.Linear(emb_dim, emb_dim, bias=bias_qkv))
            #self.proj_k = SpectralNorm(nn.Linear(emb_dim, emb_dim, bias=bias_qkv))
            #self.proj_v = SpectralNorm(nn.Linear(emb_dim, emb_dim, bias=bias_qkv))
            self.out_attention = SpectralNorm(self.out_attention)
        
        else:
            self.proj_q = EqualizedLinear(emb_dim, emb_dim, bias=bias_qkv)
            self.proj_k = EqualizedLinear(emb_dim, emb_dim, bias=bias_qkv)
            self.proj_v = EqualizedLinear(emb_dim, emb_dim, bias=bias_qkv)  
            #self.in_proj_qkv = nn.Linear(emb_dim, 3*emb_dim, bias=bias_qkv)
        
    
    def forward(self, x):
        #x: B,N,E + cls_token -> N = L+1
        #return: B,N,E
        
        B, N, E = x.shape
        H = W = int(math.sqrt(N - 1))  #H is equal to W, images H==W
                    
        
        if self.discriminator:
            cls_token, x = torch.split(x, [1, N - 1], dim = 1)
            #Same weight for Q and K
            x = x.transpose(1,2).view(B, E, H, W)
            q_k = self.proj_qk(x).flatten(start_dim=2).transpose(1, 2)#B, N, E
            v   = self.proj_v(x).flatten(start_dim=2).transpose(1, 2)#B, N, E
            
            q_k = torch.cat((cls_token, q_k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
            #qk = self.proj_qk(x)
            #k = self.proj_k(x)
            #v = self.proj_v(x)

            Q = q_k.view(B,N ,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            K = q_k.view(B,N ,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            V = v.view(B,N ,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            attn_weights = (-torch.cdist(Q, K, p=2) * self.scale).softmax(dim=-1) #Min distance, great probability
        
        else:
            #If not discriminator
            #q,k,v = self.in_proj_qkv(x).chunk(3, dim=-1)#B,N,E for All
            q = self.proj_q(x)
            k = self.proj_k(x)
            v = self.proj_v(x)
            
            Q = q.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            K = k.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            V = v.view(B,N,self.num_heads,self.head_dim).transpose(1,2)#B,Nhead,N,head_dim
            attn_weights = (torch.matmul(Q, K.transpose(-1, -2)) * self.scale).softmax(dim=-1)
        
        attn_weights = self.drop_attention(attn_weights)
        attn_out = (torch.matmul(attn_weights,V)).transpose(1, 2).reshape(B, N, E)
        attn_out = self.out_attention(attn_out)
        attn_out = self.drop_out(attn_out)
        
        return attn_out


#Improved Spectral Normalization.
#Spectral Normalization for Generative Adversarial Networks, arXiv:1802.05957
#Large Scale GAN Training for High Fidelity Natural Image Synthesis, arXiv:1809.11096
#Spectral Normalization from https://arxiv.org/abs/1802.05957
class SpectralNorm(nn.Module):
    
    def __init__(self, module, name='weight', n_power_iterations=1, eps = 1e-12):
        super(SpectralNorm, self).__init__()
        #------------------------------------------------------------------------------------------
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.init_spectral = None
        #------------------------------------------------------------------------------------------
        assert n_power_iterations >= 1, "The number of power iterations should be positive integer"
        #------------------------------------------------------------------------------------------ 
        weight = module._parameters[name]
        with torch.no_grad():
            weight_mat = weight.flatten(start_dim = 1)
            out_features, in_features = weight_mat.size()
            u = normalize(weight.new_empty(out_features).normal_(0, 1), dim=0, eps=self.eps)
            v = normalize(weight.new_empty(in_features).normal_(0, 1), dim=0, eps=self.eps)
        #------------------------------------------------------------------------------------------ 
        delattr(module, name)
        module.register_parameter(name + "_orig", weight)
        setattr(module, name, weight.data)
        module.register_buffer(name + "_u", u)
        module.register_buffer(name + "_v", v)
        #------------------------------------------------------------------------------------------ 
        for _ in range(10):
            self.spectral_norm()
        #------------------------------------------------------------------------------------------         
    def spectral_norm(self):
        weight = getattr(self.module, self.name + "_orig")
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        weight_mat = weight.flatten(start_dim = 1)
        #------------------------------------------------------------------------------------------
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = normalize(torch.mv(weight_mat.t(), u), dim=0 ,eps=self.eps, out=v)
                u = normalize(torch.mv(weight_mat, v), dim=0 ,eps=self.eps, out=u)
            #------------------------------------------------------------------------------------------
            u = u.clone(memory_format=torch.contiguous_format)
            v = v.clone(memory_format=torch.contiguous_format)
            #------------------------------------------------------------------------------------------
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        #------------------------------------------------------------------------------------------
        if self.init_spectral is None:
            self.init_spectral = sigma.detach()
        #------------------------------------------------------------------------------------------   
        setattr(self.module, self.name, self.init_spectral * weight / sigma)

    def forward(self, x):
        self.spectral_norm()
        return self.module(x)

class SLN(nn.Module):
    def __init__(self, emb_dim):#Not use Bias Affine transformation
        super().__init__()
        self.layernorm = nn.LayerNorm(emb_dim)
        self.gamma = EqualizedLinear(emb_dim, emb_dim, bias = False)
        self.beta = EqualizedLinear(emb_dim, emb_dim, bias = False)
       
    def forward(self, hidden, w):
        #w=MLP(z)-> [batch_size, emb_dim]
        #z: 2-D tensor with shape [batch_size, latent_dim]
        gamma = self.gamma(w).unsqueeze(1) # if we take batch_size x D we must add 1 to have [batch_size,1,D] then the broadcast is made automaticly
        beta = self.beta(w).unsqueeze(1)
        return gamma * self.layernorm(hidden) + beta
        
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = True, bias_init = 0, lr_mul = 1.):
        super(EqualizedLinear, self).__init__()
        self.scale = (1 / math.sqrt(in_features)) * lr_mul
        self.weight = nn.Parameter(torch.randn(out_features,in_features).div_(lr_mul))
        self.lr_mul = lr_mul
        if bias:
            self.bias = nn.Parameter(torch.ones(out_features) * bias_init)

        else:
            self.register_parameter('bias', None)

    def forward(self, x): 
      if self.bias is not None:  
        x = F.linear(x, self.weight * self.scale , bias=self.bias * self.lr_mul)
        return x
      return F.linear(x, self.weight * self.scale , bias=self.bias )

class MappingNetwork(nn.Module):
    #Mapping StyleGan2
    def __init__(self, latent_dim, emb_dim, hidden_dim = None, n_layers = 4, lr_mul = 0.01):
        super(MappingNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim or emb_dim 
        #------------------------------------------------------------------
        layers = []
        inSize =  latent_dim
        for i in range(n_layers - 1):
            layers.append(EqualizedLinear(inSize, self.hidden_dim, lr_mul = lr_mul))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) 
            inSize = self.hidden_dim  
        layers.append(EqualizedLinear(inSize, self.emb_dim, lr_mul = lr_mul ))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) #When we put it in the affin transformation    
        self.model = nn.Sequential(*layers)
        #------------------------------------------------------------------
    def forward(self, z):
        # Map z to w
        # Normalize z
        # z : Batch , Latent_dim
        # w : Batch , emb_dim
        return self.model(F.normalize(z, dim = -1))




