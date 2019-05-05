from loss import GANTrainer
from discriminator import Discriminator
from generator import Generator
from torch.utils import data
from dataset import ImageDataset
from utils import pretrain_transform, un_normalize
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #networks
    print('creating models ...')
    G = Generator()
    F = Generator()
    Dx = Discriminator()
    Dy = Discriminator()
    
    print('loading datasets ...')
    dataset_X = ImageDataset(root='train2014', transform=pretrain_transform())
    dataset_Y = ImageDataset(root='train2014', transform=pretrain_transform())
    
    print('creating trainer ...')
    trainner = GANTrainer(G, F, Dx, Dy, dataset_X, dataset_Y)
    
    print('training starts ...')
    trainner.train()