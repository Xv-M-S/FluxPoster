from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor, ne

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
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepareForText, unpack
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model2,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    get_lora_rank,
    load_checkpoint,
    InternViTWrapper
)

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class XFluxTextPipeline:
    def __init__(self, model_type, device, offload: bool = False , weight_type = torch.bfloat16):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        self.internvit = InternViTWrapper(device = "cuda:1")
        self.ae = load_ae(model_type, device="cpu")
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu")
        else:
            self.model = load_flow_model2(model_type, device="cpu")
        self.ae.to(self.device).to(weight_type)
        self.model.to(self.device).to(weight_type)

    def __call__(self,
                 image_prompt: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: torch.bfloat16 = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: torch.bfloat16 = 3,
                 timestep_to_start_cfg: int = 0,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)


        return self.forward(
            image_prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs
        )

    def forward(
        self,
        image_prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            inp_cond = prepareForText(img=x)
            last_hidden_state, pooler_output = self.internvit(image_prompt)
            # 结果移动到cuda:0
            bs = x.shape[0]
            last_hidden_state, pooler_output = last_hidden_state.to("cuda:0"), pooler_output.to("cuda:0")
            inp_cond['txt'] = last_hidden_state
            inp_cond['txt_ids'] = torch.zeros(bs, last_hidden_state.shape[1], 3).to("cuda:0")
            inp_cond['vec'] = pooler_output

            # 生成negtive image
            neg_image_prompt = Image.new("RGB", (width, height), "white")
            neg_last_hidden_state, neg_pooler_output = self.internvit(neg_image_prompt)
            neg_last_hidden_state, neg_pooler_output = neg_last_hidden_state.to("cuda:0"), neg_pooler_output.to("cuda:0")
            neg_txt = neg_last_hidden_state
            neg_txt_ids = torch.zeros(bs, neg_last_hidden_state.shape[1], 3).to("cuda:0")
            neg_vec = neg_pooler_output

            x = denoise(
                self.model,
                **inp_cond,
                timesteps=timesteps,
                guidance=guidance,
                timestep_to_start_cfg=timestep_to_start_cfg,
                true_gs=true_gs,
                neg_txt=neg_txt,
                neg_txt_ids=neg_txt_ids,
                neg_vec=neg_vec
            )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.bfloat16(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()


class XFluxTextSampler(XFluxTextPipeline):
    def __init__(self, internvit, ae, model, device):
        self.internvit = internvit
        self.ae = ae
        self.model = model
        self.model.eval()
        self.device = device
        self.offload = False
