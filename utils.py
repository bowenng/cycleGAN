from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

def pretrain_transform():
    return transforms.Compose([transforms.Resize(512),
                               transforms.CenterCrop((256,256)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])


def un_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    takes a tensor, 
    undo normalization given mean and std,
    transform to numpy,
    transpose dimension from CHW to HWC
    """
    mean = torch.FloatTensor(mean).view(1,3,1,1)
    std = torch.FloatTensor(std).view(1,3,1,1)
    
    image = tensor.cpu().detach()
    image = image*std+mean
    image = image.numpy()
    
    image = np.transpose(image, (0,2,3,1))
    
    #print(np.max(image))
    #print(np.min(image))
    return image

    
    
    

