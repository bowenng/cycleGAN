from torch.utils.data import Dataset
import os
from PIL import Image


class ImageDataset(Dataset):
    """
    dataset for images
    """
    
    def __init__(self, root, transform=None):
        self.root = root
        files = os.listdir(root)
        self.image_files = [file for file in files if self.is_image(file)]# store image file names in a list
        self.transform = transform
    
    @staticmethod
    def is_image(file):
        return file.lower().endswith(('.jpg', '.png', '.jpeg'))
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_full_path = os.path.join(self.root, self.image_files[idx])
        img = Image.open(image_full_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
        
    

