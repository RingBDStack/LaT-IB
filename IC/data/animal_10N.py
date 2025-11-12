import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch

class Animal_10N(Dataset):
    def __init__(self, mode=1, root='~/animal-10N', transform=None): 
        root = os.path.expanduser(root)
        self.transform = transform
        if mode == 0:
            txt_file = 'animal10n-train.pt'
        else:
            txt_file = 'animal10n-test.pt'
        data_path = os.path.join(root, txt_file)
        data_read = torch.load(data_path)

        self.images, self.targets = data_read.tensors
        self.targets = self.targets.squeeze(1)
        self.train_noisy_labels = self.targets

    def __len__(self):
        return len(self.targets)
 
    def __getitem__(self, index):
        targets = self.targets[index].long()
        image = self.images[index].permute(2, 0, 1)
        image = self.transform(image)
        return image, targets, index
 