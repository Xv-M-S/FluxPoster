# README

# Introduction

The README for v2 branch, v2 branch has the follow features.

* train poster generation
* inference poster generation

# Prepare for environment

```shell
conda env create -f environment.yaml
```

# Usage

## train with flux for poster generation

```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file "./accelerate/config.yaml" train_flux_poster.py --config "/home/sxm/flux-workspace/FluxPoster/train_configs/mb.yaml"
```

## inference poster generation

```shell
python3 main_poster.py \
 --use_controlnet --model_type flux-dev \
 --width 512 --height 512  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4 \
 --local_path /home/sxm/flux-workspace/FluxPoster/saves_poster_rough/checkpoint-100000/controlnet.bin
```

```shell
python3 main_poster2.py  --use_controlnet --model_type flux-dev  --width 512 --height 512  --timestep_to_start_cfg 1  --num_steps 25 --true_gs 4 --guidance 4  --local_path /home/sxm/flux-workspace/FluxPoster/saves_poster_two/checkpoint-20000/controlnet.bin --save_path ./inference_res
```
