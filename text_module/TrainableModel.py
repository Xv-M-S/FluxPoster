import torch

class TrainableModel(torch.nn.Module):
    def __init__(self, controlnet, font_embedding, auxiliary, attention_pooling):
        super().__init__()
        self.controlnet = controlnet
        self.font_embedding = font_embedding
        self.auxiliary = auxiliary
        self.attention_pooling = attention_pooling

    def forward(self, *args, **kwargs):
        return self.controlnet(*args, **kwargs)  # 这里只需返回主模型输出即可