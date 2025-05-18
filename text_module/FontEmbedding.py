from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from pre_process.recognizer import crop_image, TextRecognizer, create_predictor
from easydict import EasyDict as edict
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.TextEncoders import FrozenCLIPEmbedderT3, FrozenCLIPEmbedder
from transformers import CLIPTokenizer


def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"
    return tokens[0, 1]

def get_recog_emb(encoder, img_list):
    repeat = 3 if img_list[0].shape[1] == 1 else 1
    _img_list = [(img.repeat(1, repeat, 1, 1)*255)[0] for img in img_list]
    encoder.predictor.eval() # 启用eval模式，提取Glyph特征是不反传梯度
    _, preds_neck = encoder.pred_imglist(_img_list, show_debug=False)
    return preds_neck


def get_style_emb(encoder, img_list):
    repeat = 3 if img_list[0].shape[1] == 1 else 1
    _img_list = [(img.repeat(1, repeat, 1, 1)*255)[0] for img in img_list]
    # encoder.predictor.train() # 启用train模式，提取font特征是反传梯度
    _, preds_neck = encoder.pred_imglist(_img_list, show_debug=False)
    return preds_neck

# img: CHW, result: CHW tensor 0-255
# 用于将输入的图像张量 img 等比例缩放到指定的高度 imgH 和宽度 imgW，并用黑色填充不足的部分。
def resize_img(img, imgH, imgW):

    c, h, w = img.shape
    if h > w * 1.2:
        img = torch.transpose(img, 1, 2).flip(dims=[1])
        h, w = img.shape[1:]

    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = torch.nn.functional.interpolate(
        img.unsqueeze(0),
        size=(imgH, resized_w),
        mode='bilinear',
        align_corners=True,
    )
    padding_im = torch.zeros((c, imgH, imgW), dtype=torch.float32).to(img.device)
    padding_im[:, :, 0:resized_w] = resized_image[0]
    return padding_im

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class EncodeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeNet, self).__init__()
        chan = 16
        n_layer = 4  # downsample

        self.conv1 = conv_nd(2, in_channels, chan, 3, padding=1)
        self.conv_list = nn.ModuleList([])
        _c = chan
        for i in range(n_layer):
            self.conv_list.append(conv_nd(2, _c, _c*2, 3, padding=1, stride=2))
            _c *= 2
        self.conv2 = conv_nd(2, _c, out_channels, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        for layer in self.conv_list:
            x = self.act(layer(x))
        x = self.act(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class FontEmbedding(nn.Module):
    def __init__(
        self,
        # CLIPeEncoder配置
        version="./models/clip-vit-large-patch14",
        use_vision=False,
        img_concat_text=False,
        token_dim = 768,
        # Text Embedding Module：Glyph Encoder
        emb_type = "ocr",
        big_linear=True,
        glyph_channels=20,
        # Text Embedding Module：Position Encoder
        position_channels=1,
        # Text Embedding Module：FontStyle Encoder
        style_channels=1,
        #  Text Embedding Module：Color Encoder
        color_fourier_encode=False,
        color_big_linear=False,
        color_small_linear=True,
        device="cuda",
    ):
        super().__init__()   # 父类初始化

        # using Stable Diffusion's CLIP encoder
        tokenizer = FrozenCLIPEmbedderT3(version=version, use_vision=use_vision, img_concat_text=img_concat_text) 
        
        # 为placeholder添加新的tokenizer
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.get_token_for_string = partial(get_clip_token_for_string, clip_tokenizer)
        
        # Glyph Encoder
        self.text_predictor = create_predictor().eval()
        args = edict()
        args.rec_image_shape = "3, 48, 320"
        args.rec_batch_num = 6
        args.rec_char_dict_path = './ocr_recog/ppocr_keys_v1.txt'
        args.use_fp16 = False
        args.device = device
        self.recog = TextRecognizer(args, self.text_predictor)
        for param in self.text_predictor.parameters():
            param.requires_grad = False
        self.paddle_glyph_encoder = partial(get_recog_emb, self.recog)

        # 使用PaddleOCR提取的中间特征
        if emb_type == 'ocr':
            if big_linear:
                self.proj = nn.Sequential(
                                linear(40*64, 1280),
                                nn.SiLU(),
                                linear(1280, token_dim),
                                nn.LayerNorm(token_dim)
                                )
            else:
                self.proj = nn.Sequential(
                                zero_module(linear(40*64, token_dim)),
                                nn.LayerNorm(token_dim)
                                )
        if emb_type == 'conv':
            self.glyph_encoder = EncodeNet(glyph_channels, token_dim)


        # PositionEncoder
        # By utilizing ξp, we introduce spatial information for each text line, enabling the embedding to achieve spatial awareness.
        self.position_encoder = EncodeNet(position_channels, token_dim)

        # FontEncoder
        # self.style_encoder = EncodeNet(style_channels, token_dim)
        self.font_predictor = create_predictor()
        args = edict()
        args.rec_image_shape = "3, 48, 320"
        args.rec_batch_num = 6
        args.rec_char_dict_path = './ocr_recog/ppocr_keys_v1.txt'
        args.use_fp16 = False
        args.device = device
        self.style_encoder = TextRecognizer(args, self.font_predictor)
        for param in self.font_predictor.parameters():
            param.requires_grad = True
        self.paddle_font_encoder = partial(get_style_emb, self.style_encoder)

        self.style_proj = nn.Sequential(
                            zero_module(linear(40*64, token_dim)),
                            nn.LayerNorm(token_dim)
                            )

        if color_fourier_encode:
            self.rgb_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.rgb_encoder = TimestepEmbedding(in_channels=256*3, time_embed_dim=token_dim)
        elif color_big_linear:
            self.color_proj = nn.Sequential(
                        zero_module(linear(3, 1280)),
                        nn.SiLU(),
                        zero_module(linear(1280, token_dim)),
                        nn.LayerNorm(token_dim)
                        )
        elif color_small_linear:
            # color_small_linear -- 论文建议使用这种方法
            self.color_proj = nn.Sequential(
                                zero_module(linear(3, token_dim)),
                                nn.LayerNorm(token_dim)
                                )
    
    def encoder_visual_text(self, visual_text_info):
        raw_img_boxes = visual_text_info['raw_img_boxes']
        raw_hint_boxes = visual_text_info['raw_hint_boxes']
        mask_img_boxes = visual_text_info['mask_img_boxes']
        mask_hint_boxes = visual_text_info['mask_hint_boxes']
        colors = visual_text_info['colors']
        position_mask = visual_text_info['position_mask']
        device = visual_text_info['device']

        # glyph encoder
        for i, mask_img in enumerate(mask_img_boxes):
            mask_img_boxes[i] = resize_img(mask_img, 48, 320) # 提取特征限制图片大小为（3，48，320）
        paddle_glyph_features = self.paddle_glyph_encoder(mask_img_boxes)
        glyph_features = self.proj(paddle_glyph_features.reshape(paddle_glyph_features.shape[0], -1))

        # position encoder
        position_features = self.position_encoder(position_mask)

        # font encoder
        # 直接用图片的边缘图会好很多
        for i,hint in enumerate(raw_hint_boxes):
            raw_hint_boxes[i] = resize_img(hint, 48, 320)
        paddle_font_features = self.paddle_font_encoder(raw_hint_boxes)
        font_features = self.style_proj(paddle_font_features.reshape(paddle_font_features.shape[0], -1))

        # color encoder
        r,g,b = colors
        # 将三元组转换为张量
        rgb_tensor = torch.tensor([r, g, b], device = device, dtype=torch.bfloat16)
        colors_features = self.color_proj(rgb_tensor)

        # 特征加和
        features = glyph_features + position_features + font_features + colors_features
    
        return features

    def replace_placeholder(self, tokenized_text, embedded_text, font_features):
        placeholder_string = "*"
        self.placeholder_token = self.get_token_for_string(placeholder_string)
        b, device = tokenized_text["input_ids"].shape[0], tokenized_text["input_ids"].device
        for i in range(b):
            idx = 0
            for j,token in enumerate(tokenized_text["input_ids"][i]):
                if token == self.placeholder_token:
                    embedded_text[i][j] = font_features[i][idx]
                    idx += 1
        return embedded_text

        






