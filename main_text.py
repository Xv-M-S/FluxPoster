import argparse
from turtle import width
from PIL import Image
import os
import torch

from src.flux.xflux_text_pipeline import XFluxTextPipeline


def main():
    # 设置配置
    width = 512
    height = 512
    guidance = 4.0
    num_images_per_prompt = 1
    num_steps = 50
    seed = 123456789
    true_gs = 3.5
    timestep_to_start_cfg = 0
    save_path = "results"
    img_prompt = "/home/sxm/data02Space/idea2-train-generation-pure-text/FluxPoster/generate_datasets/datasets/random_text_2.png"
    # 模型配置
    model_type = "flux-dev"
    lora_local_path = "/home/sxm/data02Space/idea2-train-generation-pure-text/FluxPoster/lora/checkpoint-90000/lora.safetensors"
    lora_local_path = "/home/sxm/data02Space/flux/EasyText/models/pretrain.safetensors"
    device = "cuda:0"
    offload = False
    weight_type = torch.bfloat16
    

    xflux_pipeline = XFluxTextPipeline(model_type, device, offload, weight_type)
    xflux_pipeline.set_lora(lora_local_path)

    image_prompt = Image.open(img_prompt) 

    for _ in range(num_images_per_prompt):
        result = xflux_pipeline(
            image_prompt=image_prompt,
            width=width,
            height=height,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            true_gs=true_gs,
            timestep_to_start_cfg=timestep_to_start_cfg
        )
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        ind = len(os.listdir(save_path))
        result.save(os.path.join(save_path, f"result_{ind}.png"))
        seed = seed + 1


if __name__ == "__main__":
    main()
