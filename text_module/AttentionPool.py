import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_vector = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, embeddings):  # (B, L, D)
        # 确保参数与输入的数据类型和设备一致
        device = embeddings.device
        dtype = embeddings.dtype
        if self.attention_vector.device != device or self.attention_vector.dtype != dtype:
            self.attention_vector.data = self.attention_vector.data.to(device=device, dtype=dtype)
            
        att_weights = torch.einsum('bld,d->bl', embeddings, self.attention_vector)
        att_weights = F.softmax(att_weights, dim=-1)
        weighted_emb = torch.einsum('bl,bld->bd', att_weights, embeddings)
        return weighted_emb