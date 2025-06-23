import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import random
import cv2

DEBUG = True
def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

# 由于字符串不可堆叠，所以需要自定义collate_fn
def custom_collate_fn(batch):
    # 假设每个样本是一个元组 (img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result)
    imgs = torch.stack([item[0] for item in batch], dim=0)
    mask_imgs = torch.stack([item[1] for item in batch], dim=0)
    mask_hints = torch.stack([item[2] for item in batch], dim=0)
    
    captions = [item[3] for item in batch]
    texts = [item[4] for item in batch]
    bboxes = [item[5] for item in batch]
    
    return imgs, mask_imgs, mask_hints, captions, texts, bboxes



class CustomImageDataset(Dataset):
    def __init__(self, data_dir, img_size=(512,512)):
        self.images = []
        self.labels = []
        self.masks = []
        self.captions = []
        self.titles = []
        for child_dir in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, child_dir)):
                raw_img_dir = os.path.join(data_dir, child_dir, 'rendered_text_out.png')
                mask_img_dir = os.path.join(data_dir, child_dir, 'textMask.png')
                label_dir = os.path.join(data_dir, child_dir, 'labels.txt')
                caption_dir = os.path.join(data_dir, child_dir, 'label.json')
                self.images.append(raw_img_dir)
                self.labels.append(label_dir)
                self.masks.append(mask_img_dir)
                self.captions.append(caption_dir)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet的分布
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # raw_image 处理
        img = Image.open(self.images[idx])
        # img -> tensor
        # img = torch.from_numpy((np.array(img) / 127.5) - 1)
        # img = img.permute(2, 0, 1)
        img = self.transforms(img)

        # mask_image 处理
        mask_img = Image.open(self.masks[idx])
        mask_hint = canny_processor(mask_img) # 获取边缘图
        # img -> tensor
        # mask_img = torch.from_numpy((np.array(mask_img) / 127.5) - 1)
        # mask_hint = torch.from_numpy((np.array(mask_hint) / 127.5) - 1)
        # mask_img = mask_img.permute(2, 0, 1)
        # mask_hint = mask_hint.permute(2, 0, 1)
        mask_img = self.transforms(mask_img)
        mask_hint = self.transforms(mask_hint)

        # 获取captions
        jsf = json.load(open(self.captions[idx]))
        caption = jsf["prompt"]
        
        # 获取文本及其box
        texts = []
        bboxes = []
        with open(self.labels[idx], "r") as f:
            readlines = f.readlines()
        for line in readlines:
            if line.startswith("text:"):
                res = line.split(":")[1].strip().split(" ")
                texts.append(res[0])
                bbox = (float(res[1]), float(res[2]), float(res[3]), float(res[4]))
                bboxes.append(bbox)


        return img, mask_img, mask_hint, caption, texts, bboxes


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True, collate_fn=custom_collate_fn)


if __name__ == '__main__':
    args = {
        'data_dir': '/home/sxm/flux-workspace/FluxPoster/util/dataGeneration/datasets'
        # 不需要img_size, 后面会根据比例自适应调整
    }

    train_dataloader = loader(train_batch_size=1, num_workers=1, **args)
    for step, batch in enumerate(train_dataloader):
        img, mask_img, mask_hint, caption, texts, bboxes = batch
        print(f"img:{img.shape}")
        print(f"mask_img:{mask_img.shape}")
        print(f"mask_hint:{mask_hint.shape}")
        print(f"prompts:{caption}")
        print(f"texts:{texts}")
        print(f"bboxes:{bboxes}")
        break

