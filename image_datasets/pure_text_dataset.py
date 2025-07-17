import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        raw_img = Image.open(self.images[idx]).convert('RGB')
        w, h = raw_img.size
        # 对图像进行归一化处理，将像素值从 [0, 255] 范围缩放到 [-1, 1] 范围。
        # 这是许多生成模型（如 GAN、扩散模型）常用的输入格式。
        normalization_img = torch.from_numpy((np.array(raw_img) / 127.5) - 1)
        normalization_img = normalization_img.permute(2, 0, 1)
        return raw_img, normalization_img

def custom_collate(batch):
    # 只保留 tensor 部分
    raw_imgs = [item[0] for item in batch]  # item[1] 是归一化后的 tensor
    normalization_imgs = torch.stack([item[1] for item in batch], dim=0)
    return raw_imgs, normalization_imgs

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=custom_collate
    )
