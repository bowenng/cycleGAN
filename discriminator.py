import torch
from torch import nn
import numpy as np

class Discriminator(nn.Module):
    """
    PatchGAN discriminator
    
    """
    def __init__(self, in_channel=3, channel_mutiplier=64, n_layers=3,kernel_size=4, stride=2, padding=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        
        sequence = [nn.Conv2d(in_channel, channel_multiplier, kernel_size=kernel_size, stride=stride, padding=padding),
                   nn.LeakyReLU(0.2, True)]
        
        previous_channel_size = channel_multiplier
        
        for n in range(1, n_layers):
            in_channel_n = previous_channel_size # input channel size at layer n
            out_channel_n = min(2**n, 8) * channel_multiplier # output channel size at layer n
            previous_channel_size = out_channel_n
            
            sequence += [nn.Conv2d(in_channel_n, out_channel_n, kernel_size=kernel_size, stride=stride, padding=padding),
                        norm_layer,
                        nn.LeakyReLU(0.2,True)]
        
        sequence += [nn.Conv2d(previous_channel_size, 1, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.Sigmoid()]
        
        self.main = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.main(x)
            