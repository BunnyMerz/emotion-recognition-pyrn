import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset


class FER2013Custom(Dataset):
    def __init__(self, root, split, transform = None):
        self.height = 48
        self.width = 48
        self.split = split

        if split == 'train':
            file_name = 'train.csv'
        elif split == 'test':
            file_name = 'pbt.csv'
        elif split == 'validation':
            file_name = 'pvt.csv'
        else:
            raise Exception("No split found.")

        self.data = pd.read_csv(os.path.join(root, 'fer2013', file_name))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = np.reshape(
            np.fromstring(
                self.data.at[index, 'pixels'],
                dtype=np.uint8, sep=' '
                ),
            (self.height, self.width)
        )

        label = int(self.data.at[index, 'emotion'])

        if self.transform is not None:
            image = self.transform(image)

        return image, label