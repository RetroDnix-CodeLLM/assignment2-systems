import torch
from torch import Tensor

from math import sqrt
from einops import rearrange, einsum

from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

def Silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply softmax to the input tensor along the specified dimension.
    """
    x = x - x.max(dim=dim, keepdim=True).values
    in_dtype = x.dtype
    x = x.to(torch.float32)  # Ensure numerical stability
    exp_x = torch.exp(x)
    result = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return result.to(in_dtype)  # Return to original dtype

def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> Float[Tensor, "batch seq vocab"]:
    """
    Apply softmax with temperature scaling to the logits.
    
    Args:
        logits: The input logits tensor.
        temperature: The temperature for scaling.
    
    Returns:
        Scaled softmax probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    
    scaled_logits = logits / temperature
    return softmax(scaled_logits, dim=-1)