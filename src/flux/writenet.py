from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange

from .modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# 参考anytext2的WriteNet修改而来：WriteNetFlux(nn.Module):
class ControlNetFlux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams, controlnet_depth=2):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels # 64
        self.out_channels = self.in_channels # 64
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads # 128
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size # 3072
        self.num_heads = params.num_heads # 24
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.guided_hint_in = nn.Linear(1280, self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(controlnet_depth)
            ]
        )

        # add ControlNet blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(controlnet_depth):
            controlnet_block = nn.Linear(self.hidden_size, self.hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
        self.pos_embed_input = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.gradient_checkpointing = False
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.Conv2d(16, 16, 3, padding=1))
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        controlnet_cond: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        guided_hint: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # image 输入置为 0 
        image = torch.zeros_like(img).to(img.device)
        img = self.img_in(image) # img  (1,1024,64)

        # 文本的 canny 控制
        controlnet_cond = self.input_hint_block(controlnet_cond)
        controlnet_cond = rearrange(controlnet_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond

        # guided_hint 加入
        gb, gc, gh, gw = guided_hint.shape # (4,320,64,64)
        """
        # Step 1: 将空间维度 (64, 64) 分解为小块 (2, 32, 2, 32)
        # guided_hint -> [B, C, p_H, h_patch, p_W, w_patch]
        # 其中 p_H=2, h_patch=32; p_W=2, w_patch=32
        guided_hint = guided_hint.view(gb, gc, 2, gh/2, 2, gw/2)
        # Step 2: 重新排列成 [B, num_patches, flattened_patch_dim]
        # 即：[B, 32*32, 2*2*320] = [4, 1024, 1280]
        """
        # 使用 einops 一步到位
        guided_hint = rearrange(
            guided_hint,
            'b c (ph h) (pw w) -> b (h w) (c ph pw)',
            ph=2, pw=2, h=int(gh/2), w=int(gw/2)
        )
        guided_hint = guided_hint.to(img.device)
        guided_hint = guided_hint.bfloat16()
        img = img + self.guided_hint_in(guided_hint)

        # timesteps 设置为 0, 消除timesteps 的影响
        timesteps = torch.tensor([0]*timesteps.shape[0], device=timesteps.device)
        timesteps = timesteps.bfloat16()
        vec = self.time_in(timestep_embedding(timesteps, 256)) # vec (1,3072)

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        
        vec = vec + self.vector_in(y) # y -> clip encode prompt (1,3072)
        txt = self.txt_in(txt) # txt -> t5 encode prompt (1,512,3072)

        ids = torch.cat((txt_ids, img_ids), dim=1) # 位置编码
        pe = self.pe_embedder(ids)

        block_res_samples = ()
        for block in self.double_blocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    pe,
                )
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            block_res_samples = block_res_samples + (img,) # 这行代码的作用是将 img 添加到 block_res_samples 元组中，生成一个新的元组，并将结果重新赋值给 block_res_samples。

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        return controlnet_block_res_samples
