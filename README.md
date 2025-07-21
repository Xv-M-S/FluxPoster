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
accelerate launch --main_process_port 29586 --config_file "./accelerate/config.yaml" train_flux_text.py --config "train_configs/pure_text_lora.yaml" 
```

``` bash
export CUDA_VISIBLE_DEVICES=1
accelerate launch --main_process_port 29586 --config_file "./accelerate/config.yaml" train_flux_lora_deepspeed.py --config "train_configs/test_lora.yaml" 
```

为了加快训练，将internVit移动到cuda:1,需要申请两块gpu

``` bash
export CUDA_VISIBLE_DEVICES=2, 3
accelerate launch --main_process_port 29588 --config_file "./accelerate/config.yaml" train_flux_text.py --config "train_configs/pure_text_lora.yaml" 
```

离线运行
``` bash
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=2,3
nohup accelerate launch --main_process_port 29586 --config_file "./accelerate/config.yaml" train_flux_text.py --config "train_configs/pure_text_lora.yaml" > train_output.log 2>&1 &
```

# Inference on 4090 lora
基于lora的train只能非常小范围的微调，不适合训练一个新的模型。

``` bash
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=2,3
python main_text.py 
```

# train samll flux on 4090 for pure text

``` bash
export CUDA_VISIBLE_DEVICES=6,7
accelerate launch --main_process_port 29568 --config_file "./accelerate/config.yaml" train_flux_text_all.py --config "train_configs/train_text.yaml" 
```

``` bash
nohup accelerate launch --main_process_port 29568 --config_file "./accelerate/config.yaml" train_flux_text_all.py --config "train_configs/train_text.yaml" > pure_text_train.log 2>&1 &
```

# inference on 4090 for flux-dev

``` bash
export CUDA_VISIBLE_DEVICES=0,1
python main_flux.py
```

# inference on 4090 for easy text

``` bash
export CUDA_VISIBLE_DEVICES=0,1
python main_easy_text.py
```

# InternViT 1.5
多模态视觉模型InternViT 1.5在文本识别相关任务上表现出出色的性能。
故采用InternViT 1.5提取视觉文本特征作为唯一的标识。

# 生成纯文本数据集
纯文本数据集生成方法：直接在纯白的图片上使用现有的渲染技术渲染出文字。
``` bash
cd ./generate_datasets
python3 generate.py 
```

# train on single 汉字

``` bash
export CUDA_VISIBLE_DEVICES=4,5
nohup accelerate launch --main_process_port 29530 --config_file "./accelerate/config.yaml" train_flux_text.py --config "train_configs/pure_text_lora2.yaml" > singlehanzi.log 2>&1 &
```