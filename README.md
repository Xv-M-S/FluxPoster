# introduction
it is a repo for train a model that can generate accurate pure visual text on image.

# dependency
1. './models/clip-vit-large-patch14'

# train on 4090 machine base controlnet

``` bash
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file "./accelerate/config.yaml" train_flux_poster.py --config "./train_configs/test_poster_controlnet.yaml"
```

## nohup train

``` bash
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=1
nohup accelerate launch --config_file "./accelerate/config.yaml" train_flux_poster.py --config "./train_configs/test_poster_controlnet.yaml" > train.log 2>&1 &
```

# inference controlnet

``` bash
export CUDA_VISIBLE_DEVICES=2
python3 main_poster.py \
 --use_controlnet --model_type flux-dev \
 --width 512 --height 512  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4 \
 --local_path /home/sxm/data02Space/idea2-train-generation-pure-text/FluxPoster/saves_poster/checkpoint-250/controlnet.bin
```

# train on 4090 base lora

``` bash
export CUDA_VISIBLE_DEVICES=1
accelerate launch --main_process_port 29586 --config_file "./accelerate/config.yaml" train_flux_text.py --config "train_configs/test_lora.yaml" 
```

``` bash
export CUDA_VISIBLE_DEVICES=1
accelerate launch --main_process_port 29586 --config_file "./accelerate/config.yaml" train_flux_lora_deepspeed.py --config "train_configs/test_lora.yaml" 
```

# InternViT 1.5
多模态视觉模型InternViT 1.5在文本识别相关任务上表现出出色的性能。
故采用InternViT 1.5提取视觉文本特征作为唯一的标识。