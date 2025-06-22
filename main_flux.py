import argparse
from PIL import Image
import os

from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor
import math

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







def main():
    ## 处理参数
    device = torch.device("cuda")
    offload = False
    model_type = "flux-dev"
    seed = 86
    height, width = 512, 512
    guidance = 4
    model_type = "flux-dev"
    num_steps = 50
    batch_size = 1
    data_dtype = torch.bfloat16
    weight_dtype = torch.bfloat16
    save_path = "/home/sxm/flux-workspace/FluxPoster/inference_res"

    # 推理参数
    image_prompts = "a pair of glasses"

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

    # 设置噪声
    x = get_noise(
        1, height, width, device=device,
        dtype=torch.bfloat16, seed=seed
    )
    timesteps = get_schedule(
        num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=True,
    )
    torch.manual_seed(seed)
    with torch.no_grad():
        if offload:
            t5, clip = t5.to(device), clip.to(device)
        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=image_prompts)
        img = inp_cond["img"]

       

        i = 0
        # this is ignored for schnell
        guidance_vec = torch.full((x.shape[0],), guidance, device=x.device, dtype=x.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)
            pred = model(
                img=img,
                img_ids=inp_cond["img_ids"],
                txt=inp_cond["txt"],
                txt_ids=inp_cond["txt_ids"],
                y=inp_cond['vec'],
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            img = img + (t_prev - t_curr) * pred
            i += 1

        x = img
        x = unpack(x.float(), height, width).to(data_dtype)
        x = vae.decode(x)

    x1 = x.clamp(-1, 1)
    x1 = rearrange(x1[-1], "c h w -> h w c")
    output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
    output_img.save("/home/sxm/flux-workspace/FluxPoster/inference_res/result_0.png")

if __name__ == "__main__":
    main()
