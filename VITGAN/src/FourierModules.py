import torch
from torch import nn
import math
import torch.nn.functional as F


class ModulatedLinear(nn.Module):
    def __init__(self, in_features, out_features, demodulation=True):
        super(ModulatedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(in_features,out_features))        
        self.demodulation = demodulation


    def forward(self, Efou, y):
        #Efou ->(BxL,PxP,E)
        #y -> (B, L, E)
        #groups in conv1d apply in each num_peatches
        #Weight ->(out_features, in_features)
        #y -> (batch_size, 1, self.in_features)
        batch_size = Efou.shape[0]
        y = y.view(batch_size, self.in_features,1) #(BxL, in_features, 1)
        weight = self.scale * self.weight * y

        if self.demodulation:
            #dconf = torch.rsqrt(weight.pow(2).sum(dim = 1) + 1e-8)
            weight = weight * torch.rsqrt(weight.pow(2).sum(dim = 1) + 1e-8).view(batch_size, 1, self.out_features) # batch, out_features, in_features

        return torch.matmul(Efou, weight)
#Positional Embedding Each positional embedding of ViT networks is a linear projection of patch position followed by a sine activation function. 
#The patch positions are normalized to lie between −1.0 and 1.0.
#arXiv:2006.09661 Super interesent article.
#Activation Sine
class Sine(nn.Module):
    def __init__(self,w_0=30):
        super(Sine, self).__init__()
        self.w_0=w_0

    def forward(self, x):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        # arXiv:2006.09661
        return torch.sin(self.w_0 * x)

class SineLinear(nn.Module):
    def __init__(self, in_features, out_features, w_0 = 30, c = 6., is_first = False, bias = True, output_linear = False):
        #By default w_0=30
        #c=6 if not the first layer
        #see article arXiv:2006.09661
        #Super Effiecent
        super(SineLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features    
        self.is_first = is_first
        self.w_0 = w_0
        self.c = c
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = Sine(w_0 = w_0) if not output_linear else None
        self.output_linear = output_linear

        self.weights_init() #Else pytorch initialization (refered in articles) but without sqrt

    def weights_init(self):
        if self.is_first : 
            nn.init.uniform_(self.linear.weight.data,( -1 / self.in_features), (1 / self.in_features))
        else:
            nn.init.uniform_(self.linear.weight.data,-math.sqrt(self.c / self.in_features) / self.w_0, math.sqrt(self.c / self.in_features) / self.w_0)

    def forward(self, x):
        output = self.linear(x)
        if self.output_linear:
            return output
        return self.activation(output)        

class SineModulatedLinear(nn.Module):
    def __init__(self, in_features, out_features, w_0 = 30, c = 6., is_first = False,  output_linear = False, demodulation=True):
        #By default w_0=30
        #c=6 if not the first layer
        #see article arXiv:2006.09661
        #Super Effiecent
        super(SineModulatedLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features    
        self.is_first = is_first
        self.w_0 = w_0
        self.c = c
        self.linear = ModulatedLinear(in_features, out_features, demodulation=demodulation)
        self.activation = Sine(w_0 = w_0) if not output_linear else None
        self.output_linear = output_linear

        self.weights_init() #Else pytorch initialization (refered in articles) but without sqrt

    def weights_init(self):
        if self.is_first :
            nn.init.uniform_(self.linear.weight.data,( -1 / self.in_features), (1 / self.in_features))
        else:
            nn.init.uniform_(self.linear.weight.data,-math.sqrt(self.c / self.in_features) / self.w_0, math.sqrt(self.c / self.in_features) / self.w_0)

    def forward(self, x, y):
        output = self.linear(x, y)
        if self.output_linear:
            return output
        return self.activation(output)

#To overcome the detrimental effect of traditional non-linearities like ReLU/tanh on modeling fine details and higher-order derivative of the input signals, 
#Sitzmann et al. proposes to use sinusoidal activation functions that allow explicit supervision on any derivatives of the input signal.
class SIREN(nn.Module):
    def __init__(self, in_features, out_features = 3 , hidden_layer = None, bias=True, n_layers=4, w_0=30., c = 6., output_linear=False):
        super(SIREN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.hidden_layer = hidden_layer or in_features
        self.w_0 = w_0

        layers = ([SineLinear(self.in_features, self.hidden_layer, w_0 = self.w_0, c=c ,is_first = True, bias = bias)]
                  + [SineLinear(self.hidden_layer, self.hidden_layer, w_0 = self.w_0, c=c,bias = bias) for _ in range(n_layers - 1)]
                  + [SineLinear(self.hidden_layer, self.out_features, w_0 = self.w_0, c=c,bias = bias, output_linear=output_linear)])
        
        self.linear_maps = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.linear_maps(x)

class SIRENModulated(nn.Module):
    def __init__(self, in_features, out_features = 3 , hidden_layer = None, n_layers=4, w_0=30., c = 6., output_linear=False , demodulation = True):
        super(SIRENModulated, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.hidden_layer = hidden_layer or in_features
        self.w_0 = w_0
        layers = ([SineModulatedLinear(self.in_features, self.hidden_layer, w_0 = self.w_0, c=c ,is_first = True, demodulation = demodulation)]
                  + [SineModulatedLinear(self.hidden_layer, self.hidden_layer, w_0 = self.w_0, c=c, demodulation = demodulation) for _ in range(n_layers - 1)]
                  + [SineModulatedLinear(self.hidden_layer, self.out_features, w_0 = self.w_0, c=c, demodulation = demodulation, output_linear = output_linear)])
        
        self.linear_maps = nn.ModuleList(layers)
        

    def forward(self, x, y):
        for layer in self.linear_maps:
            x = layer(x , y)
        return x
        
#Fourier Features Let Networks Learn
#High Frequency Functions in Low Dimensional Domains
#arXiv:2006.10739
class FFNEmbedding(nn.Module):
    def __init__(self, in_features = 2, emb_size = 512, scale = 10.):
        #in_features is the dimension (x,y)
        super(FFNEmbedding, self).__init__()
        self.in_features = in_features
        self.emb_size = emb_size 
        self.scale = scale
        self.B = nn.Parameter(torch.randn(in_features, 512//in_features) * scale, requires_grad=False)
        self.out_features = emb_size  # (x,y) so 2 dim
    
    def forward(self, x):
        return torch.cat([torch.sin((2 * math.pi * x) @ self.B),
                          torch.cos((2 * math.pi * x) @ self.B)], dim = -1)

#Use ReLU and Sin instead of Sigmoid (in the original article they use Sigmoid as output), Sin
class FFN(nn.Module):
    def __init__(self, in_features, out_features = 3 , hidden_layer = None, emb_size = 256 ,bias=True, n_layers=4, scale=10., output_linear=False):
        super(FFN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layer = hidden_layer or in_features
        self.scale = scale
        self.output_linear = output_linear 
        self.embedder = FFNEmbedding(in_features, emb_size = emb_size, scale = scale)
        #-----------------------------------------------------------------------------------
        layers = []
        layers.append(nn.Linear(self.embedder.out_features, self.hidden_layer, bias = bias))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(self.hidden_layer, self.hidden_layer, bias = bias))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_layer, self.out_features, bias = bias))            
        #-----------------------------------------------------------------------------------
        self.linear_maps = nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.embedder(x)
        x = self.linear_maps(x)

        if self.output_linear:
            return x
        return torch.sin(x)

class FFNModulated(nn.Module):
    def __init__(self, in_features, out_features = 3 , hidden_layer = None, emb_size = 256 , n_layers=4, scale=10., output_linear=False, demodulation=True):
        super(FFNModulated, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layer = hidden_layer or in_features
        self.scale = scale
        self.output_linear = output_linear 
        self.embedder = FFNEmbedding(in_features, emb_size = emb_size, scale = scale)
        #-----------------------------------------------------------------------------------
        layers = []
        layers.append(ModulatedLinear(self.embedder.out_features, self.hidden_layer, demodulation=demodulation))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(ModulatedLinear(self.hidden_layer, self.hidden_layer, demodulation=demodulation))
            layers.append(nn.ReLU())
        layers.append(ModulatedLinear(self.hidden_layer, self.out_features, demodulation=demodulation))            
        #-----------------------------------------------------------------------------------
        self.linear_maps = nn.ModuleList(layers)

    
    def forward(self, x, y):
        x = self.embedder(x)
        for layer in self.linear_maps:
            x = layer(x, y)
        if self.output_linear:
            return x
        return torch.sin(x)
#Implicit Neural Representation for Patch Generation Each positional embedding is a linear projection of pixel coordinate followed by a sine activation 
# function (hence the name Fourier encoding). The pixel coordinates for P 2 pixels are normalized to lie between −1.0 and 1.0. The 2-layer MLP takes 
# positional embedding Efou as its input, and it is conditioned on patch embedding yi via weight modulation as in (Karras et al., 2020b; Anokhin et al., 2021).