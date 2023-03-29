
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from torch import Tensor
import numpy as np
from PIL import Image


class BrainImageDataset(Dataset):
    def __init__(self, mode: str, img_size: int, ):
        assert mode in ['train', 'test', 'val']
        print(f'Loading {mode} data...')
        self.images = []
        if mode == 'train':
            for i in range(len(train_files)):
                print(i)
                self.images.append(np.asarray(Image.open(
                    train_files[i]).resize((128, 128), Image.ANTIALIAS)))
        elif mode == 'test':
            for i in range(len(test_files)):
                self.images.append(np.asarray(Image.open(
                    test_files[i]).resize((128, 128), Image.ANTIALIAS)))
        else:
            for i in range(len(val_files)):
                self.images.append(np.asarray(Image.open(
                    val_files[i]).resize((128, 128), Image.ANTIALIAS)))
        self.images = torch.Tensor(np.array(self.images))

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        img = self.images[idx]
        return img
