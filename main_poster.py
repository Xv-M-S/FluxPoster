import argparse
from PIL import Image
import os

from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor

from einops import rearrange
import uuid
import os

from src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    ImageProjModel,
)
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    get_lora_rank,
    load_checkpoint
)

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
import torchvision.transforms as transforms

# 数据相关
from image_datasets.poster_dataset import loader
# 数据预处理
from pre_process.process import preProcess, auxiliaryPreProcess, getFontPrompt
import torchvision.transforms as transforms

# Texts相关模型导入
from text_module.FontEmbedding import FontEmbedding
from text_module.AuxiliaryLatent import AuxiliaryLatentMoudle
from text_module.AttentionPool import AttentionPooling
from text_module.TrainableModel import TrainableModel

def convert_tensors_to_bfloat16(data, device):
    if isinstance(data, dict):
        return {key: convert_tensors_to_bfloat16(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_tensors_to_bfloat16(item, device) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device = device, dtype=torch.bfloat16)
    else:
        return data  # 保留非张量数据不变


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, required=False,
        help="The input text prompt"
    )
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--img_prompt", type=str, default=None,
        help="Path to input image prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0,
        help="Strength of input image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "zoe", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    return parser


def main(args):
    ## 处理参数
    device = torch.device(args.device)
    offload = args.offload
    model_type = args.model_type
    image_encoder_path = "openai/clip-vit-large-patch14"
    seed = 86
    height, width = 512, 512
    guidance = args.guidance
    model_type = "flux-dev"
    local_path = args.local_path
    num_steps = 50
    batch_size = 2
    data_dtype = torch.bfloat16
    weight_dtype = torch.bfloat16
    save_path = "/home/sxm/flux-workspace/FluxPoster/inference_res"

    ## 加载数据
    args = {
    'raw_img_dir': './posterData/input',
    'mask_img_dir': './posterData/mask',
    'label_dir': './posterData/label',
    'img_size': (height, width)
    }
    test_dataloader = loader(train_batch_size=2, num_workers=4, **args)

    ## 加载模型
    # 加载ae模型
    vae = load_ae(model_type, device="cpu" if offload else device)
    vae.to(data_dtype)
    # 加载文本编码模型
    clip = load_clip(device)
    t5 = load_t5(device, max_length=512)
    # 加载flux-dev模型
    model = load_flow_model(model_type, device="cpu" if offload else device)
    model.to(device)
    # 加载controlnet模型 -- 经过了预训练的controlnet模型
    controlnet = load_controlnet(model_type, device).to(data_dtype)
    checkpoint = load_checkpoint(local_path, "repo_id", "name")
    controlnet.load_state_dict(checkpoint, strict=False)
    controlnet.to(device)

    
    # 添加其他模块 - text_Embedding
    font_embedding = FontEmbedding(device = device).to(device).to(data_dtype)
    auxiliary = AuxiliaryLatentMoudle().to(device).to(data_dtype)
    attention_pooling = AttentionPooling(hidden_dim=768).to(device).to(data_dtype)

    for step, batch in enumerate(test_dataloader):
        bs_img, bs_hint, bs_mask_img, bs_mask_hint, bs_raw_caption, bs_caption, bs_ocr_result = batch

        # 获取控制图片、图片提示词、文本提示词、字体嵌入、辅助latent
        control_image = bs_hint.to(device) # 在writeNet中没有使用，而是构建了一个全零的大小相等的变量
        image_prompts = bs_caption
        text_prompts, texts = [], []

        font_features = [] # text embedding
        guided_hints = []

        for img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result in zip(bs_img, bs_hint, bs_mask_img, bs_mask_hint, bs_raw_caption, bs_caption, bs_ocr_result):
            img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result = img.squeeze(0), hint.squeeze(0), mask_img.squeeze(0),mask_hint.squeeze(0),raw_caption,caption,ocr_result
            text_info = auxiliaryPreProcess(img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result)
            if text_info is None:
                guided_hints.append(torch.zeros(1,320,64,64).to(device=device, dtype=weight_dtype))
            else:
                # Ensure all inputs are bfloat16
                text_info = convert_tensors_to_bfloat16(text_info, device)
                # Convert glyphs to grayscale if it's RGB (C=3)
                if text_info['glyphs'].shape[0] == 3:
                    transform = transforms.Grayscale()
                    text_info['glyphs'] = transform(text_info['glyphs'].unsqueeze(0)).squeeze(0)
                
                guided_hint = auxiliary.encode(text_info)
                guided_hints.append(guided_hint)

            # 提取text的prompt
            text_prompt, text = getFontPrompt(ocr_result) 
            text_prompts.append(text_prompt)
            texts.append(text)


            visual_text_info = preProcess(img, hint, mask_img, mask_hint, raw_caption, caption, ocr_result)
            visual_text_info = convert_tensors_to_bfloat16(visual_text_info, device)
            if visual_text_info is None:
                features = []
                font_features.append(features)
            else:
                # [x, 768], 其中x为图片中文本的个数
                visual_text_info['device'] = device
                print(f"visual_text_info: {visual_text_info['mask_img_boxes'][0].dtype}")
                features = font_embedding.encoder_visual_text(visual_text_info)
                font_features.append(features)

        guided_hints_batch = torch.cat(guided_hints, dim = 0) # auxiliary feature
        text_prompts = [prompt if isinstance(prompt, str) and prompt.strip() != "" else "[PAD]" for prompt in text_prompts]

        
        # 原始噪声数据
        # x = get_noise(
        #     batch_size, height, width, device=device,
        #     dtype=torch.bfloat16, seed=seed
        # )
        x = torch.randn(
            batch_size,
            3,
            height,
            width,
            device=device,
            dtype=weight_dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        x = vae.encode(x)

        print(f"prompts: {len(image_prompts)}")
        print(f"x:{x.shape}")

        # timesteps生成
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        # timesteps = torch.repeat_interleave(timesteps, repeats=batch_size, dim=0)

        print(f"timesteps:{timesteps}")

        # 模型加载
        torch.manual_seed(seed)
        with torch.no_grad():
            print(f"prepare {x.shape}")
            inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=image_prompts)
            x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            # x = inp_cond['img']
            # neg_inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=neg_prompt)
            text_pooler, text_hidden, tokenized_text = clip(text_prompts, detail=True)
            replace_text_hidden = font_embedding.replace_placeholder(tokenized_text, text_hidden, font_features)
            y_text = attention_pooling(replace_text_hidden)

            guidance_vec = torch.full((x.shape[0],), guidance, device = device, dtype=x.dtype)

            print(f"y text: {y_text.shape}")
            print(f"guidance vec: {guidance_vec.shape}")
            
            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                weight_dtype = x.dtype
                # controlnet 推理
                t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
                print(f"t vec: {t_vec.shape}")
                print(f"inp_cond['img_ids']: {inp_cond['img_ids'].shape}")
                print(f"inp_cond['txt_ids']: {inp_cond['txt_ids'].shape}")
                print(f"inp_cond['txt']: {inp_cond['txt'].shape}")

                # 将img_ids 和 txt_ids 转换为bfloat16
                # inp_cond = convert_tensors_to_bfloat16(inp_cond, device)
                # block_res_samples = controlnet(
                #     img=x,
                #     img_ids=inp_cond['img_ids'].to(weight_dtype),
                #     controlnet_cond=control_image.to(weight_dtype),
                #     txt=inp_cond['txt'].to(weight_dtype),
                #     txt_ids=inp_cond['txt_ids'].to(weight_dtype),
                #     y=y_text,
                #     timesteps=t_vec.to(weight_dtype),
                #     guidance=guidance_vec.to(weight_dtype),
                #     guided_hint = guided_hints_batch.to(weight_dtype)
                # )

                pred = model(
                    img=x,
                    img_ids=inp_cond['img_ids'].to(weight_dtype),
                    txt=inp_cond['txt'].to(weight_dtype),
                    txt_ids=inp_cond['txt_ids'].to(weight_dtype),
                    y=inp_cond['vec'].to(weight_dtype),
                    timesteps=t_vec,
                    # block_controlnet_hidden_states=[
                    #     sample.to(dtype=weight_dtype) for sample in block_res_samples
                    # ],
                    guidance=guidance_vec.to(weight_dtype),
                )
                x = x + (t_prev - t_curr) * pred
            to_pil = transforms.ToPILImage()

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            ind = len(os.listdir(save_path))
            x = rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=2, pw=2, h=32, w=32)
            result = vae.decode(x).cpu().float()
            bs = result.shape[0]
            for i in range(bs):
                x1 = result[i]
                x1 = x1.clamp(-1, 1)
                x1 = rearrange(x1, "c h w -> h w c")
                output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
                # pil_img =  to_pil(result[i])
                ind = len(os.listdir(save_path))
                output_img.save(os.path.join(save_path, f"result_{ind}.png"))


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
