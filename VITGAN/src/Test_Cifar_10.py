
from Discriminator import *
from Generator import *
from utils import *
from FourierModules import *
from Modules import *
import torchvision
from torch_ema import ExponentialMovingAverage 
from DiffAugment import *
from torchvision import transforms
from torch import nn
import torch
from torch.nn.functional import normalize #v=\frac{v}{\max (\|v\|_{p}, \epsilon)}
import torch.nn.functional as F
import math
import torchvision
from torch_ema import ExponentialMovingAverage
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = (10,6)

to_pil = transforms.ToPILImage()
renorm = transforms.Normalize((-1.), (2.))



if __name__ == "__main__":
    latent_dim = 128
    torch.cuda.empty_cache()
    netG = Generator(emb_dim = 300,latent_dim = latent_dim, num_heads = 5, n_blocks = 3, mlp_ratio = 4,image_size=32, demodulation=False, out_features=3).cuda()
    netD = Discriminator(emb_dim = 300, num_heads = 5, n_blocks = 3, mlp_ratio = 4,image_size=32, in_channels=3).cuda()
    #netG.apply(weights_init)
    #netD.apply(weights_init)
    ########################################################################################################################
    acc = 0
    for param in netG.parameters():
        acc+=param.numel()
    print("Generator",acc)
    acc = 0
    for param in netD.parameters():
        acc+=param.numel()
    print("Discriminator",acc)
    ########################################################################################################################
    batch_size = 32
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0., 0., 0.), (1., 1., 1.))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    iteration = (len(trainset)//batch_size)
    ########################################################################################################################
    g_lr = 0.002
    d_lr = 0.002
    beta1 = 0.0 
    beta2 = 0.99
    lam = 0.7
    lambda_real = 10.
    lambda_fake = 10.
    gamma = 0.999
    gen_optimizer = torch.optim.Adam(netG.parameters(), g_lr, (beta1, beta2))
    dis_optimizer = torch.optim.Adam(netD.parameters(),d_lr, (beta1, beta2))
    #gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma, last_epoch=- 1, verbose=False)
    ema = ExponentialMovingAverage(netG.parameters(), decay=0.995)
    fixed_z = torch.FloatTensor(np.random.normal(0, 1, (20, latent_dim))).cuda()

    nb_epoch = 200 
    display_freq = 500
    #criterion = nn.BCELoss().cuda()
    #############################################################################################################


    g_losses = []
    d_losses = []


    j = 0


    for epoch in range(nb_epoch):
        
        pbar = tqdm(enumerate(trainloader))
        for i, batch in pbar:
            im, labels = batch 
            im = im.cuda()
            im_augmented = DiffAugment(im, policy='color,translation,cutout', channels_first=True)
            cur_batch_size = im.shape[0]
            z = torch.randn(cur_batch_size, latent_dim).cuda()
            #######################################################################################
            zero_label= torch.zeros(cur_batch_size).cuda()# Size-> (cur_batch_size,1)
            one_label = torch.ones(cur_batch_size).cuda()# Size-> (cur_batch_size,1)
            
            
            fake_image=netG(z)
            augmented_fake = DiffAugment(fake_image, policy='color,translation,cutout', channels_first=True).detach()
        
            ###########################################################################################
            yhat_real = netD(im_augmented).flatten()
            real_loss = nn.BCEWithLogitsLoss()(yhat_real,one_label)
            bcr_real = F.mse_loss(yhat_real, netD(im).flatten())
            
            yhat_fake=netD(augmented_fake).flatten()
            fake_loss=nn.BCEWithLogitsLoss()(yhat_fake,zero_label)
            bcr_fake = F.mse_loss(yhat_fake, netD(fake_image).flatten())
            ############################################################################################
            ###
            ### Discriminator
            ###
            d_loss = ((real_loss + fake_loss) / 2) + bcr_real * lambda_real + bcr_fake * lambda_fake
            
            d_loss =(real_loss+fake_loss)/2 #+ lam*l2_reg#     YOUR CODE HERE
            #########################################################################################################################
            dis_optimizer.zero_grad()
            d_loss.backward(retain_graph = True) # we need to retain graph=True to be able to calculate the gradient in the g backprop
            #torch.nn.utils.clip_grad_norm_(netD.parameters(), 5.)
            dis_optimizer.step()

            
            ###
            ### Generator
            ###
        
            
            g_loss = nn.BCEWithLogitsLoss()(netD(fake_image).flatten(),one_label) #      YOUR CODE HERE
            gen_optimizer.zero_grad()
            g_loss.backward()
            #torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.)
            gen_optimizer.step()
            ema.update()
            
            # Save Metrics
            
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            
            avg_real_score = yhat_real.sigmoid().mean().item()
            avg_fake_score = yhat_fake.sigmoid().mean().item()

            pbar.set_description(f"it: {j}; g_loss: {g_loss}; d_loss: {d_loss}; avg_real_score: {avg_real_score}; avg_fake_score: {avg_fake_score}")
            
            if i % display_freq == 0:
                fake_im = netG(fixed_z).cpu().detach()
                
                un_norm = renorm(fake_im) # for visualization
                
                grid = torchvision.utils.make_grid(un_norm, nrow=5)
                pil_grid = to_pil(grid)  
                
                plt.imshow(pil_grid)
                plt.savefig('comparison.png')
                plt.show()
                
                
                plt.plot(range(len(g_losses)), g_losses, label='g loss')
                plt.plot(range(len(g_losses)), d_losses, label='d loss')
                
                plt.legend()
                plt.show()
                
            j += 1
            
            del z
            del yhat_real
            del yhat_fake
            del fake_image
            del zero_label
            del one_label
            del im
            del g_loss
            del d_loss
            del real_loss
            del fake_loss
        print(f"Iteration {epoch}")    
        #gen_scheduler.step()