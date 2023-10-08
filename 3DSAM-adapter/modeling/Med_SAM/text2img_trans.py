import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Type
import math
from einops import rearrange
from torch.utils.checkpoint import checkpoint

class Text2ImageTransformer(nn.Module):
    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 384,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.align_layers = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList()
        self.mlahead_channels=128
        for _ in range(depth):
            self.layers.append(
                Text2ImageAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate
                )
            )

    def forward(self, image_embedding, text_embedding) -> Tuple[Tensor, Tensor]:
        # image_embedding = [1,256,32,32,32]
        # text_embedding = [1,512]
        image_embedding = rearrange(image_embedding, 'b c d h w -> b (d h w) c')
        text_embedding = self.align_layers(text_embedding).unsqueeze(1)
        # image_embedding = [1,32768,256]
        # text_embedding = [1,1,256]
        for i, layer in enumerate(self.layers): # 2個block
            print(f"loop: {i}")
            image_embedding = layer(
                image_embedding,
                text_embedding
            )

        # image_embedding包含了圖像和文本之間的關聯資訊。
        image_embedding = rearrange(image_embedding, 'b (d h w) c -> b c d h w', d=32, h=32, w=32)
        print(f"mask decoder result: {image_embedding.size()}")
        return image_embedding
    
class Text2ImageAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 384,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        # self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn = Attention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        # self.norm3 = nn.LayerNorm(embedding_dim)
        # self.global_query = nn.parameter.Parameter(data=0.1 * torch.randn(1, 10, embedding_dim)) # [1,10,256]的可訓練參數(隨機產生的正態分布*0.1)

    def forward(self, image_embedding, text_embedding) -> Tensor:
        # # self attention
        # # self_out = self.self_attn(q=image_embedding, k=image_embedding, v=image_embedding)
        # self_out = checkpoint(self.self_attn, image_embedding, image_embedding, image_embedding)
        # self_out = self.norm1(image_embedding + self_out) # skip connection
        # print("self_out", self_out.size())

        # # cross attention
        # # cross_out = self.cross_attn(q=self_out, k=text_embedding, v=text_embedding)
        # cross_out = checkpoint(self.cross_attn, self_out, text_embedding, text_embedding)
        # cross_out = self.norm2(self_out + cross_out) # skip connection
        # print("cross_out", self_out.size())

        cross_out = checkpoint(self.cross_attn, image_embedding, text_embedding, text_embedding)
        cross_out = self.norm1(image_embedding + cross_out) # skip connection

        # MLP block
        mlp_out = checkpoint(self.mlp, cross_out)
        mlp_out = self.norm2(cross_out + mlp_out) # skip connection

        return mlp_out

class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate # 全聯接層的縮放大小
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads 必須整除 embedding_dim"

        # MLP dim => internal_dim
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)

        # MLP internal_dim => dim
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q) # MLP
        k = self.k_proj(k) # MLP
        v = self.v_proj(v) # MLP

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head) # 這種縮放操作可以防止當特徵數值很大時，點積的結果過大導致的梯度爆炸問題。
        attn = torch.softmax(attn, dim=-1) # 在最後一個維度做，對應於一個Q對所有K的相似度

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out
    
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            act(),
            nn.Linear(mlp_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)