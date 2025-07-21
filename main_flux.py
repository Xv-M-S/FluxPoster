import argparse
from PIL import Image
import os
import torch

from src.flux.xflux_pipeline_4090 import XFluxPipeline

def main():
    # 设置配置
    width = 512
    height = 512
    guidance = 4.0
    num_images_per_prompt = 1
    num_steps = 50
    seed = 1234
    true_gs = 3.5
    timestep_to_start_cfg = 0
    save_path = "results"
    img_prompt = "/home/sxm/data02Space/idea2-train-generation-pure-text/FluxPoster/generate_datasets/datasets/random_text_2.png"
    # 模型配置
    model_type = "flux-dev"
    lora_local_path = "/home/sxm/data02Space/idea2-train-generation-pure-text/FluxPoster/lora/checkpoint-90000/lora.safetensors"
    device0 = "cuda:0"
    device1 = "cuda:1"
    offload = False
    weight_type = torch.bfloat16
    prompt = "you are"

    xflux_pipeline = XFluxPipeline(model_type, device0, device1, offload)

    for _ in range(num_images_per_prompt):
        result = xflux_pipeline(
            prompt=prompt,
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
