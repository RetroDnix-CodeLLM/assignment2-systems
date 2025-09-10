import torch
from torch import Tensor

from math import sqrt
from einops import rearrange, einsum

from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

from cs336_basics.model_utils import softmax

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        """
        super(Linear, self).__init__()
        
        w = torch.empty(out_features, in_features, device=device, dtype=dtype) # w = in out
        theta = sqrt(2.0 / (in_features + out_features))
        torch.nn.init.normal_(w, mean=0.0, std=theta)

        b = torch.zeros(out_features, device=device, dtype=dtype) # b = out
        self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(b)

    def forward(self, x: Float [Tensor,"... d_in"]) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out') + self.bias

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. 
        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel  
            device: torch.device | None = None Device to store the parameters on  
            dtype: torch.dtype | None = None Data type of the parameters  
        """
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.normal_(self.weight, mean=0.0, std=1)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. 
        Args:
            d_model: int Hidden dimension of the model  
            eps: float = 1e-5 Epsilon value for numerical stability  
            device: torch.device | None = None Device to store the parameters on  
            dtype: torch.dtype | None = None Data type of the parameters  
        """
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) # shape: (batch_size, sequence_length, 1)
        result = (x / rms) * self.weight.to(torch.float32)
        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff:int, device=None, dtype=None):
        """
        Construct a SwiGLU module.
        """
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SwiGLU activation function.
        """

        w1x = self.w1(x) # shape: (..., d_model) -> (..., d_ff)
        sigu_w1x = w1x * torch.sigmoid(w1x)
        w3x = self.w3(x) # shape: (..., d_model) -> (..., d_ff)
        return self.w2(sigu_w1x * w3x) # shape: (..., d_ff) -> (..., d_model)
    
class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, dtype=None, device=None):
        """
        Rotary Positional Embedding (RoPE) 模块。

        Args:
            theta: 控制频率分布的参数，通常为 10000
            d_k: 每个 head 的维度（必须为偶数）
            max_seq_len: 最大序列长度
            device: 使用的设备
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k 必须为偶数才能使用 RoPE"

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.theta = theta

        # 创建旋转频率：shape [d_k // 2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # 生成 [max_seq_len, d_k // 2]
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        freqs = pos * inv_freq.unsqueeze(0)  # [max_seq_len, d_k // 2]

        self.register_buffer("cos", torch.cos(freqs).to(dtype), persistent=False)  # [max_seq_len, d_k // 2]
        self.register_buffer("sin", torch.sin(freqs).to(dtype), persistent=False)

    def forward(
        self, 
        x: Float[Tensor, " ... seq_len d_k"], 
        token_positions: Int[Tensor, " ... seq_len"]
    ) -> torch.Tensor:
        """
        将旋转位置编码应用到输入向量上。

        Args:
            x: 输入张量，形状为 [..., seq_len, d_k]
            token_positions: 每个 token 的位置索引，形状为 [..., seq_len]

        Returns:
            应用 RoPE 后的张量，形状不变。
        """
        # 获取 cos/sin: shape = [..., seq_len, d_k // 2]
        cos = self.cos[token_positions]  # [..., seq_len, d_k // 2]
        sin = self.sin[token_positions]

        # 拆分 x 为偶数和奇数维
        x1 = x[..., ::2]  # [..., seq_len, d_k//2]
        x2 = x[..., 1::2]

        # 应用二维旋转
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)  # [..., seq_len, d_k//2, 2]

        # 合并最后两个维度为 d_k
        return x_rotated.flatten(-2)

def scaledDotProductAttention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"]|None = None
)-> Float[Tensor, " ... queries d_v"]:
    """
    Compute the scaled dot product attention.
    
    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., keys, d_v)
        mask: Mask tensor of shape (..., queries, keys)
    
    Returns:
        Attention output tensor of shape (..., queries, d_v)
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys') / sqrt(d_k)
    
    # 应用因果mask
    if mask is not None:
        # 将 mask 为 0 的位置设为 -1e9，使 softmax 后趋近于0
        scores = scores.masked_fill(mask == 0, float('-1e9'))
    
    attn_weights = softmax(scores, dim=-1).to(V.dtype)  # (..., queries, keys)
    return einsum(attn_weights, V, '... queries keys, ... keys d_v -> ... queries d_v')

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        """
        Aegs:
            d_model: int Dimensionality of the Transformer block inputs.  
            num_heads: int Number of heads to use in multi-head self-attention.
        """
        super(MultiHeadSelfAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.theta = theta
        self.max_seq_len = max_seq_len

        self.Wqkv = Linear(d_model, d_model * 3, device=device, dtype=dtype)

        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, "... sequence_length d_in"], 
        token_positions: Int[Tensor, " ... sequence_length"] | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
            mask: 可选，形状为 (batch, seq_len, seq_len) 或 (batch, 1, seq_len, seq_len)
        Returns:
            输出张量，形状为 (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model, f"Expected d_model {self.d_model}, but got {d_model}"

        qkv = self.Wqkv(x)  # (batch, seq_len, d_model) * (d_model * 3, d_model) -> (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)  # (batch, seq_len, 3, num_heads, d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        if token_positions is not None:
            # 如果提供了 token_positions，则应用 RoPE
            rope = RoPE(theta=self.theta, d_k=self.d_k, max_seq_len=self.max_seq_len, device=x.device)
            Q = rope(Q, token_positions)  # (batch, num_heads, seq_len, d_k)
            K = rope(K, token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = ~mask  # 下三角为True
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, self.num_heads, -1, -1)

        # 调用 scaled dot-product attention
        attn_output = scaledDotProductAttention(Q, K, V, mask)  # (batch, num_heads, seq_len, d_k)

        # 合并多头：(batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        # 最后输出线性层
        return self.Wo(attn_output)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        """
        Construct a Transformer block.
        Args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            d_ff: int Dimensionality of the feed-forward network.
        """
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... sequence_length d_in"]) -> torch.Tensor:
        """
        Apply the Transformer block to the input tensor.
        """
        token_positions = torch.arange(x.size(1), device=x.device)

        x = x + self.attn(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        """
        Construct a Transformer language model.
        Args:
            vocab_size: int Size of the vocabulary.
            d_model: int Dimensionality of the Transformer block inputs.
            n_layers: int Number of Transformer blocks.
            n_heads: int Number of heads to use in multi-head self-attention.
            d_ff: int Dimensionality of the feed-forward network.
        """
        super(TransformerLM, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta=rope_theta, max_seq_len=context_length, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "... sequence_length"]) -> torch.Tensor:
        """
        Apply the Transformer language model to the input tensor.
        """
        x = self.embedding(x)  # (batch_size, sequence_length) -> (batch_size, sequence_length, d_model)
        
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # (batch_size, sequence_length, d_model)
        return self.output(x)  # (batch_size, sequence_length, vocab_size)