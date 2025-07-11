# introduction
it is a repo for train a model that can generate accurate pure visual text on image.

# dependency
1. './models/clip-vit-large-patch14'

# run

``` bash
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file "./accelerate/config.yaml" train_flux_poster.py --config "./train_configs/test_poster_controlnet.yaml"
```

# nohup run

``` bash
export CUDA_VISIBLE_DEVICES=1
nohup accelerate launch --config_file "./accelerate/config
.yaml" train_flux_poster.py --config "./train_configs/test_poster_controlnet.yaml" > train.log 2>&1 &
```