import sys
sys.path.append("/home/sxm/flux-workspace/x-flux-main")
from image_datasets.poster_dataset import loader
from pre_process.process import preProcess
from pre_process.FontEmbedding import FontEmbedding
import numpy as np
from PIL import Image

def visual_info(visual_text_info):
    raw_img_boxes = visual_text_info['raw_img_boxes']
    raw_hint_boxes = visual_text_info['raw_hint_boxes']
    mask_img_boxes = visual_text_info['mask_img_boxes']
    mask_hint_boxes = visual_text_info['mask_hint_boxes']
    colors = visual_text_info['colors']
    position_mask = visual_text_info['position_mask']

    visual_dir = "/home/sxm/flux-workspace/x-flux-main/pre_process/visual"

    for i in range(len(raw_img_boxes)):
        raw_image_box = raw_img_boxes[i] * 255
        raw_hint_box = raw_hint_boxes[i] * 255
        mask_image_box = mask_img_boxes[i] * 255
        mask_hint_box = mask_hint_boxes[i] * 255
        # color = colors[i]
        position_mask_box = position_mask[i] * 255

        # 转换为 (H, W, C) 格式
        raw_image_box = raw_image_box.permute(1, 2, 0).numpy().astype(np.uint8)
        raw_hint_box = raw_hint_box.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_image_box = mask_image_box.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_hint_box = mask_hint_box.permute(1, 2, 0).numpy().astype(np.uint8)
        position_mask_box = position_mask_box.permute(1, 2, 0).numpy().astype(np.uint8)

        # 保存图像
        Image.fromarray(raw_image_box).save(f'{visual_dir}/{i}_raw_image_box.png')
        Image.fromarray(raw_hint_box).save(f'{visual_dir}/{i}_raw_hint_box.png')
        Image.fromarray(mask_image_box).save(f'{visual_dir}/{i}_mask_image_box.png')
        Image.fromarray(mask_hint_box).save(f'{visual_dir}/{i}_mask_hint_box.png')

        # 对于位置掩码（灰度图），需要先转换为 RGB 格式
        position_mask_box = np.repeat(position_mask_box, 3, axis=2)
        Image.fromarray(position_mask_box).save(f'{visual_dir}/{i}_position_mask_box.png')




if __name__ == "__main__":
    args = {
        'raw_img_dir': './posterDataSets/input',
        'mask_img_dir': './posterDataSets/mask',
        'label_dir': './posterDataSets/label',
        'img_size': (512,512)
    }

    font_embedding = FontEmbedding()

    train_dataloader = loader(train_batch_size=2, num_workers=1, **args)
    for step, batch in enumerate(train_dataloader):
        bs_img, bs_hint, bs_mask_img, bs_mask_hint, bs_raw_caption, bs_caption, bs_ocr_result = batch
        # train_dataloader给所有数据都加了一个维度,所以需要去掉第一个维度
        for img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result in zip(bs_img, bs_hint, bs_mask_img, bs_mask_hint, bs_raw_caption, bs_caption, bs_ocr_result):
            img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result = img.squeeze(0), hint.squeeze(0), mask_img.squeeze(0),mask_hint.squeeze(0),raw_caption,caption,ocr_result
            visual_text_info = preProcess(img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result)
            if visual_text_info is None:
                features = None
            else:
                # [x, 768], 其中x为图片中文本的个数
                features = font_embedding.encoder_visual_text(visual_text_info)
                print(f"features: {features.shape}")
                visual_info(visual_text_info)
