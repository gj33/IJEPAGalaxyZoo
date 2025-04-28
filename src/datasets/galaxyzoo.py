import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import torchvision
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

_GLOBAL_SEED = 0
logger = getLogger()

dir_cat = "/content/drive/MyDrive/GalaxyZoo/"
dir_image = '/content/ijepa/data/GalaxyZoo/images_gz2/images'
df = pd.read_csv(dir_cat+'gz2_train.csv')

class GalaxyZooDataset(Dataset):
    '''Galaxy Zoo 2 image dataset
        Args:
            dataframe : pd.dataframe, outputs from the data_split function
                e.g. df_train / df_valid / df_test
            dir_image : str, path where galaxy images are located
            label_tag : str, class label system to be used for training
                e.g. label_tag = 'label1' / 'label2' / 'label3' / 'label4'
    '''

    def __init__(self, dataframe, dir_image, label_tag='label1', transform=None):
        self.df = dataframe
        self.transform = transform
        self.dir_image = dir_image
        self.label_tag = label_tag

    
    def __getitem__(self, index):
        galaxyID = self.df.iloc[[index]].galaxyID.values[0]
        file_img = os.path.join(self.dir_image, str(galaxyID) + '.jpg')
        image = Image.open(file_img)

        if self.transform:
            image = self.transform(image)
        
        label = self.df.iloc[[index]][self.label_tag].values[0]

        return image, label

    def __len__(self):
        return len(self.df)

def make_galaxyzoo(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    if transform is None:
        # If no transform provided, at least convert images to Tensors
        transform = transforms.ToTensor()
    dir_image = '/content/ijepa/data/GalaxyZoo/images_gz2/images'
    dataset = GalaxyZooDataset(df, dir_image, label_tag='label1', transform=transform)
    logger.info('ImageNet dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler