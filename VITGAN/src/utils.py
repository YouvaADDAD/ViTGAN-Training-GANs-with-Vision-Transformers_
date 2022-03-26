from torch import nn
import torch
import math
import collections

def input_to_tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

def requires_grad(model, grad=True):
    for param in model.parameters():
        param.requires_grad = grad

def normalizeNeg1And1(size):
    vector = torch.arange(size)
    return 2 * ((vector - (size-1))/(size - 1)) - 1 

def generate_coords(patch_size, dim = 2):
    """
    size : int, the size of coordinate
    dim = 2 represent (x, y) 
    """
    coordinates = tuple(dim * [torch.linspace(-1, 1, patch_size)])
    mgrid = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def init_weights_normal(module):
    if type(module) == nn.Linear:
        if hasattr(module, 'weight'):
            nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity='relu', mode='fan_out')

def init_weights_selu(module):
    if type(module) == nn.Linear:
        if hasattr(module, 'weight'):
            num_input = module.weight.size(-1)
            nn.init.normal_(module.weight, std=1/math.sqrt(num_input))

def init_weights_elu(module):
    if type(module) == nn.Linear:
        if hasattr(module, 'weight'):
            num_input = module.weight.size(-1)
            nn.init.normal_(module.weight, std=math.sqrt(1.5505188080679277)/math.sqrt(num_input))

def init_weights_xavier(module):
    if type(module) == nn.Linear:
        if hasattr(module, 'weight'):
            nn.init.xavier_normal_(module.weight)

def init_weights_uniform(module):
    if type(module) == nn.Linear:
        if hasattr(module, 'weight'):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

def sine_init(module, w0=30.):
    with torch.no_grad():
        if hasattr(module, 'weight'):
            num_input = module.weight.size(-1)
            module.weight.uniform_(-math.sqrt(6/num_input)/w0, math.sqrt(6/num_input)/w0)

def first_layer_sine_init(module):
    with torch.no_grad():
        if hasattr(module, 'weight'):
            num_input = module.weight.size(-1)
            module.weight.uniform_(-1/num_input, 1/num_input)

class PositionalEmbedding1D(nn.Module):

    def __init__(self, d_model, max_len=4096):
        super(PositionalEmbedding1D,self).__init__()

        positional_emb = torch.zeros(max_len, d_model).float()
        positional_emb.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        positional_emb[:, 0::2] = torch.sin(position * div_term)
        positional_emb[:, 1::2] = torch.cos(position * div_term)

        positional_emb = positional_emb.unsqueeze(0)
        self.register_buffer('positional_emb', positional_emb)

    def forward(self, x):
        return self.positional_emb[:, :x.size(1)]

class Conv3x3(nn.Module):
    def __init__(self, emb_dim, kernel_size=3):
        #N, C_in, H_in, W_in 
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels=emb_dim, out_channels=2*emb_dim, kernel_size=kernel_size, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        return x 

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

