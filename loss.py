import torch
from torch import nn
from torch.utils import data
from torch import optim
import os

class GANTrainer():
    """
    Full objective that combines adversial loss and cycle consistency loss
    """
    
    def __init__(self, G, F, Dx, Dy, dataset_X, dataset_Y, epochs=100, batch_size=1, num_workers=1, lr=0.0002,device='cuda',check_progress_every=100, save_every=100):
        self.G = G.to(device)
        self.F = F.to(device)
        self.Dx = Dx.to(device)
        self.Dy = Dy.to(device)
        
        self.optimizer_G = optim.Adam(G.parameters(), lr=lr)
        self.optimizer_Dy = optim.Adam(Dy.parameters(), lr=lr)
        self.optimizer_F = optim.Adam(F.parameters(), lr=lr)
        self.optimizer_Dx = optim.Adam(Dx.parameters(), lr=lr)
        
        self.epochs = epochs
        self.device = device
        self.check_progress_every = check_progress_every
        self.save_every = save_every
        
        self.dataloader_X = data.DataLoader(dataset_X, batch_size=batch_size, num_workers=num_workers)
        self.dataloader_Y = data.DataLoader(dataset_Y, batch_size=batch_size, num_workers=num_workers)
        
        self.adversial_loss_G = AdversialLoss().to(device)
        self.adversial_loss_F = AdversialLoss().to(device)
        
        self.adversial_loss_Dx = AdversialLoss().to(device)
        self.adversial_loss_Dy = AdversialLoss().to(device)
        
        self.cycle_consistency_loss_x = CycleConsistencyLoss().to(device)
        self.cycle_consistency_loss_y = CycleConsistencyLoss().to(device)

        
    def train(self):
        step = 0
        for e in range(self.epochs):
            for x, y in zip(self.dataloader_X, self.dataloader_Y):
                step += 1
                
                x = x.to(self.device)
                y = y.to(self.device)
                #generate G(x) F(G(x)) F(y) G(F(y))
                fake_y = self.G(x)
                restored_x = self.F(fake_y)
                fake_x = self.F(y)
                restored_y = self.G(fake_x)
                
                #discriminate Dy(G(x)) and Dx(F(y))
                #train G and F
                self.G.zero_grad()
                self.F.zero_grad()
                
                
                adv_loss_G = self.adversial_loss_G(self.Dy(fake_y), True)
                adv_loss_F = self.adversial_loss_F(self.Dx(fake_x), True)
                cyc_loss_x = self.cycle_consistency_loss_x(x, restored_x)
                cyc_loss_y = self.cycle_consistency_loss_y(y, restored_y)
                
                #Full objective = adversial loss + lambda(=10) * cycle consistency loss
                generator_loss = 0.5*(adv_loss_G + adv_loss_F) + 10 * (cyc_loss_x + cyc_loss_y)
                
                generator_loss.backward()
                #update G and F
                self.optimizer_G.step()
                self.optimizer_F.step()
                
                #train Dx and Dy
                self.Dx.zero_grad()
                self.Dy.zero_grad()
                
                adv_loss_Dx = self.adversial_loss_Dx(self.Dx(fake_x.detach()), False) + self.adversial_loss_Dx(self.Dx(x), True)
                adv_loss_Dx.backward()
                self.optimizer_Dx.step()
                
                adv_loss_Dy = self.adversial_loss_Dy(self.Dy(fake_y.detach()), False) + self.adversial_loss_Dy(self.Dy(y), True)
                adv_loss_Dy.backward()
                self.optimizer_Dy.step()
                
                if step % self.check_progress_every == 0:
                    print('Loss:')
                    print('\tGenerator: {}'.format(generator_loss.item()))
                    print('\tDy: {}'.format(adv_loss_Dy.item()))
                    print('\tDx: {}'.format(adv_loss_Dx.item()))
                
                if step % self.save_every == 0:
                    model = {'G': self.G.state_dict(),
                            'F' : self.F.state_dict(),
                            'Dy' : self.Dy.state_dict(),
                            'Dx' : self.Dx.state_dict(),
                            'step': step}
                    torch.save(model, os.path.join('saved_models','{}.pth'.format(step)))
    
    
class AdversialLoss(nn.Module):
    """
    based on LSGAN
    ref: https://arxiv.org/abs/1611.04076
    """
    
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.FloatTensor([real_label]))
        self.register_buffer('fake_label', torch.FloatTensor([fake_label]))
        self.loss = nn.MSELoss()
        
    def forward(self, prediction, is_real_image):
        """
        calculate the MSELoss between prediction and target
        NOTE: real_label if is_real_image==True else use fake_label
        """
        
        if is_real_image:
            target = self.real_label.expand(prediction.size())
        else:
            target = self.fake_label.expand(prediction.size())
            
        return self.loss(prediction, target)
    
    
class CycleConsistencyLoss(nn.Module):
    """
    L1 Loss between x and F(G(x))
    ref: https://arxiv.org/pdf/1703.10593.pdf
    """
    
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, x, restored_x):
        return self.loss(x, restored_x)
    
        
        
        
        