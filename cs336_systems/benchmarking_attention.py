import torch
import time
import itertools
import pandas as pd

from cs336_basics import scaledDotProductAttention
from cs336_systems import FlashAttnV2Torch, FlashAttnV2Triton
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

class naive_attention_warpper():
    @staticmethod
    def apply(*args, **kwargs):
        return scaledDotProductAttention(*args, **kwargs)

# class torch_attention_warpper():
#     @staticmethod
#     def apply(*args, **kwargs):
#         with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
#             attn_output = scaled_dot_product_attention(*args, **kwargs)
#         return attn_output

def benchmark(attn_method, method_name):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    # d_models = [16, 32, 64, 128]
    d_models = [128, ]
    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384]
    num_warmup = 5
    num_iters = 100

    for d_model, seq_len in itertools.product(d_models, seq_lens):
        Wq = torch.randn(d_model, d_model, device=device, requires_grad=True)
        Wk = torch.randn(d_model, d_model, device=device, requires_grad=True)
        Wv = torch.randn(d_model, d_model, device=device, requires_grad=True)

        # Warmup
        for _ in range(num_warmup):
            input = torch.randn(batch_size, seq_len, d_model, device=device)
            Q = input @ Wq
            K = input @ Wk
            V = input @ Wv
            attn_method.apply(Q, K, V)

        # Forward timing

        input = torch.randn(batch_size, seq_len, d_model, device=device)
        Q = input @ Wq
        K = input @ Wk
        V = input @ Wv
        torch.cuda.reset_peak_memory_stats(device)

        start = time.time()
        for _ in range(num_iters):
            attn_method.apply(Q, K, V)
        torch.cuda.synchronize()
        forward_time = time.time() - start

        # Measure memory before backward
        mem_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # Backward timing
        # start = time.time()
        # for _ in range(num_iters):
        #     input = torch.randn(batch_size, seq_len, d_model, device=device)
        #     target = torch.randn(batch_size, seq_len, d_model, device=device)
        #     Q = input @ Wq
        #     K = input @ Wk
        #     V = input @ Wv
        #     output = attn_method.apply(Q, K, V)
        #     loss = torch.nn.functional.mse_loss(output, target)
        #     loss.backward()
        
        # torch.cuda.synchronize()
        # backward_time = time.time() - start

        print(f"{method_name},{d_model},{seq_len},{forward_time:.4f},{mem_used:.2f}")

attn_methods = [
    (naive_attention_warpper, "Naive Attention"),
    #(torch_attention_warpper, "PyTorch Built-in"),
    #(FlashAttnV2Torch, "FlashAttention v2 Torch"),
    (FlashAttnV2Triton, "FlashAttention v2 Triton"),
]


if __name__ == "__main__":
    print("method_name", "d_model", "seq_len", "forward_time(s)", "mem(MB)", sep=",")
    for attn_method, method_name in attn_methods:
        benchmark(attn_method, method_name)
        
