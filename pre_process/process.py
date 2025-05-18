import ast
import logging
import numpy as np
import math
import re
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cropImage(image, bboxes):
    _, height, width = image.shape

    result = []
    for box in bboxes:
        x_center, y_center, w, h = box
        x_center  = x_center * width
        y_center  = y_center * height
        w = w * width
        h = h * height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        try:
            cropped_image = image[:, y1:y2, x1:x2]  # 通道维度保持不变
            result.append(cropped_image)
        except Exception as e:
            print(f"Error cropping box {box}: {e}")

    return result

"""
我们采用一种非学习的方法来创建一个颜色选择器来获取文本的RGB标签。
首先，对文本区域内所有像素的颜色进行聚类和排序，从中选择最上面的主色块。
"""
def color_picker(raw_img_boxes, mask_img_boxes):
    # 提取字体的颜色
    colors = []

    for raw_image, mask_image in zip(raw_img_boxes, mask_img_boxes):
        """
            根据掩码图像从原始图像中提取掩码区域的像素值，并计算平均颜色。
            
            参数:
            - raw_image: 原始图像的路径或 PIL.Image 对象。
            - mask_image: 掩码图像的路径或 PIL.Image 对象。
            
            返回:
            - avg_color: 掩码区域的平均颜色，格式为 (R, G, B)
        """
        # 确保原始图像和掩码图像大小一致
        if raw_image.shape != mask_image.shape:
            print(f"raw_image size: {raw_image.shape}, mask_image size: {mask_image.shape}")
            raise ValueError("原始图像和掩码图像的大小必须一致")
        
        # 将原始图像和掩码图像转换为 NumPy 数组
        raw_array = np.array(raw_image)
        mask_array = np.array(mask_image)
        
        # 提取掩码区域的像素值
        # 假设掩码图像的白色区域 (255) 表示掩码区域
        mask_pixels = raw_array[mask_array == 255]
        
        # 如果掩码区域为空，则返回默认颜色（如黑色）
        if mask_pixels.size == 0:
            return (0, 0, 0)
        
        # 计算掩码区域的像素值的平均值
        avg_color = np.mean(mask_pixels.reshape(-1, 3), axis=0)
        
        # 将平均颜色转换为整数并返回
        colors.append(tuple(avg_color.astype(int)))
    return colors

def generate_position_mask(bboxes, width, height):
    position_mask = []
    for box in bboxes:
        x_center, y_center, w, h = box
        x_center = x_center * width
        y_center = y_center * height
        w = w * width
        h = h * height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # 创建一个形状为 (height, width) 的 numpy 数组
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        # 将 numpy 数组转换为 PyTorch 张量
        mask_tensor = torch.from_numpy(mask/127.5 - 1).unsqueeze(0)  # 添加通道维度
        mask_tensor = mask_tensor.float()
        position_mask.append(mask_tensor)

    # 将列表中的所有张量堆叠成一个张量
    position_mask = torch.stack(position_mask, dim=0)
    return position_mask

def convert_tensors_to_bfloat16(data):
    if isinstance(data, dict):
        return {key: convert_tensors_to_bfloat16(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_tensors_to_bfloat16(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.to(dtype=torch.bfloat16)
    else:
        return data  # 保留非张量数据不变
def preProcess(img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result):
    bboxes = []
    texts = []
    for line in ocr_result:
        bboxes.append(line[0])
        print(line)
        texts.append(line[1][0])

    if len(bboxes) == 0:
        return None
    
    # 裁剪图片和hint
    raw_img_boxes = cropImage(img, bboxes)
    raw_hint_boxes = cropImage(hint, bboxes)

    mask_img_boxes = cropImage(mask_img, bboxes)
    mask_hint_boxes = cropImage(mask_hint, bboxes)

    colors = color_picker(raw_img_boxes, mask_img_boxes)

    _, height, width = img.shape
    position_mask = generate_position_mask(bboxes, width, height)

    visual_text_info = {
        "raw_img_boxes": raw_img_boxes,
        "raw_hint_boxes": raw_hint_boxes,
        "mask_img_boxes": mask_img_boxes,
        "mask_hint_boxes": mask_hint_boxes,
        "colors": colors,
        "position_mask": position_mask,
    }

    visual_text_info = convert_tensors_to_bfloat16(visual_text_info)

    return visual_text_info

def raw_image_position_mask(bboxes, image, mask = False):
    # Get image dimensions
    _, height, width = image.shape
    if mask:
        image_copy = np.zeros((height, width), dtype=np.uint8)
    else:
        # Create a copy of the image to avoid modifying the original
        image_copy = image.clone() if isinstance(image, torch.Tensor) else np.copy(image)

    # Iterate over each bounding box
    for box in bboxes:
        x_center, y_center, w, h = box

        # Scale normalized coordinates to image dimensions
        x_center = x_center * width
        y_center = y_center * height
        w = w * width
        h = h * height

        # Calculate bounding box corners
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if mask:
            # Set the region inside the bounding box to black (0s)
            image_copy[y1:y2, x1:x2] = 255  # Assuming image shape is [C, H, W]
        else:
            # Set the region inside the bounding box to white (255s)
            image_copy[ : ,y1:y2, x1:x2] = 255

    return image_copy



def auxiliaryPreProcess(img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result):
    bboxes = []
    texts = []
    for line in ocr_result:
        bboxes.append(line[0])
        texts.append(line[1][0])

    if len(bboxes) == 0:
        return None

    glyphs_img = mask_img

    _, height, width = img.shape
    position_mask = raw_image_position_mask(bboxes, img, True)
    position_mask = torch.from_numpy((position_mask / 127.5) - 1)
    position_mask = position_mask.unsqueeze(0)
    text_mask_img = raw_image_position_mask(bboxes, img)

    text_info = {
        "glyphs": glyphs_img,
        "positions": position_mask,
        "masked_x": text_mask_img
    }

    return text_info

def modify_prompt(prompt):
    PLACE_HOLDER = '*'
    prompt = prompt.replace('“', '"')
    prompt = prompt.replace('”', '"')
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [' ']
    else:
        for s in strs:
            prompt = prompt.replace(f'"{s}"', f'{PLACE_HOLDER}', 1)
    return prompt, strs

def getFontPrompt(ocr_result):
    prompt = 'Texts are '
    texts = []
    for line in ocr_result:
        texts.append(line[1][0])
    
    if len(texts) == 0:
        return prompt, None
    elif len(texts) == 1:
        prompt += f'"{texts[0]}"'
    else:
        prompt += ' and '.join(f'"{text}"' for text in texts)
    prompt, strs = modify_prompt(prompt)
    
    return prompt, strs


if __name__ == "__main__":
    ocr_result = [
        [[0.1, 0.1, 0.2, 0.2], ['text1']],
        [[0.3, 0.3, 0.4, 0.4], ['text2']],
        [[0.5, 0.5, 0.6, 0.6], ['text3']],
    ]
    prompt,strs = getFontPrompt(ocr_result)
    print(prompt)
    print(strs)
