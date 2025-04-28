import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2


def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def resize_image(image, target_width, target_height, output_path=None):
    # 获取图片的原始宽度和高度
    original_width, original_height = image.size
    
    # 计算缩放比例
    scale_width = target_width / original_width
    scale_height = target_height / original_height
    scale = min(scale_width, scale_height)  # 选择较小的缩放比例
    
    # 计算调整后的宽度和高度
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # 调整图片大小
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 创建一个目标尺寸的黑色背景图像
    background = Image.new('RGB', (target_width, target_height), (255, 255, 255))
    
    # 计算粘贴位置，使图像居中
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # 将调整大小后的图像粘贴到背景图像上
    background.paste(resized_image, (paste_x, paste_y))
    
    # 如果需要，可以将调整后的图片保存到指定路径
    if output_path:
        background.save(output_path)
    # print(f"backgrround:{background.size}")
    
    return background

# 由于字符串不可堆叠，所以需要自定义collate_fn
def custom_collate_fn(batch):
    # 假设每个样本是一个元组 (img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result)
    imgs = torch.stack([item[0] for item in batch], dim=0)
    hints = torch.stack([item[1] for item in batch], dim=0)
    mask_imgs = torch.stack([item[2] for item in batch], dim=0)
    mask_hints = torch.stack([item[3] for item in batch], dim=0)
    
    raw_captions = [item[4] for item in batch]
    captions = [item[5] for item in batch]
    ocr_results = [item[6] for item in batch]
    
    return imgs, hints, mask_imgs, mask_hints, raw_captions, captions, ocr_results

class CustomImageDataset(Dataset):
    def __init__(self, raw_img_dir, mask_img_dir, label_dir, img_size=(512,512)):
        self.images = [os.path.join(raw_img_dir, i) for i in os.listdir(raw_img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size
        self.label_dir = label_dir
        self.mask_img_dir = mask_img_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # raw_image 处理
        img = Image.open(self.images[idx])
        img = resize_image(img, self.img_size[0], self.img_size[1])
        hint = canny_processor(img) # 获取边缘图

        img = torch.from_numpy((np.array(img) / 127.5) - 1)
        img = img.permute(2, 0, 1)
        hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
        hint = hint.permute(2, 0, 1)

        # mask_image 处理
        img_name = self.images[idx].split('/')[-1]
        base_name = img_name[:img_name.rfind('.')]
        mask_img = Image.open(os.path.join(self.mask_img_dir, base_name + '.png'))
        mask_img = resize_image(mask_img, self.img_size[0], self.img_size[1])
        mask_hint = canny_processor(mask_img) # 获取边缘图

        # print(f"mask_img.shape={torch.from_numpy((np.array(mask_img))).shape}, mask_hint.shape={torch.from_numpy((np.array(mask_hint))).shape}")

        mask_img = torch.from_numpy((np.array(mask_img) / 127.5) - 1)
        mask_hint = torch.from_numpy((np.array(mask_hint) / 127.5) - 1)
        # mask_img = mask_img.permute(2, 0, 1)
        mask_img = mask_img.unsqueeze(0)
        mask_hint = mask_hint.permute(2, 0, 1)

        # 获取标注
        jsf = json.load(open(os.path.join(self.label_dir, base_name + '.json')))
        raw_caption = jsf['description']
        caption = jsf['finalDescription']
        ocr_result = jsf['ocr_result']

        return img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True, collate_fn=custom_collate_fn)


if __name__ == '__main__':
    """
    raw_img_dir: ./posterDataSets/input
    mask_img_dir: ./posterDataSets/mask
    label_dir: ./posterDataSets/label
    """
    args = {
        'raw_img_dir': './posterDataSets/input',
        'mask_img_dir': './posterDataSets/mask',
        'label_dir': './posterDataSets/label',
        'img_size': (512,512)
    }

    train_dataloader = loader(train_batch_size=1, num_workers=1, **args)
    for step, batch in enumerate(train_dataloader):
        # img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result = batch
        img, control_image, mask_img, mask_hint, raw_caption, prompts, ocr_result = batch

