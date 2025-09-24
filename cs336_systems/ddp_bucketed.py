from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from time import time

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class _Bucket:
    params: List[nn.Parameter]
    flat_buffer: torch.Tensor
    offsets: List[Tuple[int, int]]  # (start, numel) per param
    ready_count: int = 0
    work: Optional[dist.Work] = None


class DDPBucketed(nn.Module):
    """
    Distributed Data Parallel container using gradient bucketing to improve
    communication efficiency.

    - Broadcasts parameters and buffers from rank 0 to all ranks on init.
    - Groups parameters into buckets with total size <= bucket_size_mb.
    - During backward, when all grads in a bucket are accumulated, launches
      a single async all-reduce on the bucket's flattened gradient buffer.
    - After backward, before optimizer.step(), call finish_gradient_synchronization()
      to wait for outstanding communications, average gradients by world size,
      and scatter reduced grads back into per-parameter .grad tensors.

    Notes:
    - For simplicity, we assume parameters in a bucket share device and dtype.
      If a parameter's device or dtype differs from the current bucket, a new
      bucket is started.
    - Parameters are deduplicated to handle tied/shared weights.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized. Call dist.init_process_group first.")
        self.module = module
        self._dedup_param_ids = set()
        self._time_waiting_for_comm = 0.0
        self.bucket_size_bytes = max(1, int(bucket_size_mb * 1024 * 1024))

        # Ensure all ranks start from rank 0's parameters and buffers
        self._broadcast_model_state_from_rank0()

        # Build buckets and register hooks
        self._buckets: List[_Bucket] = []
        self._build_buckets()
        self._register_param_hooks()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @property
    def parameters_iter(self) -> Iterable[nn.Parameter]:
        # Deduplicate parameters (important for tied/shared weights)
        self._dedup_param_ids.clear()
        for p in self.module.parameters(recurse=True):
            if id(p) in self._dedup_param_ids:
                continue
            self._dedup_param_ids.add(id(p))
            yield p

    def _broadcast_model_state_from_rank0(self) -> None:
        """Broadcast parameters and buffers from rank 0 to all ranks."""
        for p in self.module.parameters(recurse=True):
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers(recurse=True):
            dist.broadcast(b.data, src=0)

    def _build_buckets(self) -> None:
        """Group parameters into buckets respecting bucket_size_bytes, device, dtype."""
        current_params: List[nn.Parameter] = []
        current_offsets: List[Tuple[int, int]] = []
        current_size_bytes = 0
        current_device = None
        current_dtype = None

        def flush_bucket():
            nonlocal current_params, current_offsets, current_size_bytes, current_device, current_dtype
            if not current_params:
                return
            total_numel = sum(n for _, n in current_offsets)
            flat = torch.empty(total_numel, dtype=current_dtype, device=current_device)
            self._buckets.append(_Bucket(params=current_params.copy(), flat_buffer=flat, offsets=current_offsets.copy()))
            # reset
            current_params.clear()
            current_offsets.clear()
            current_size_bytes = 0
            current_device = None
            current_dtype = None

        for p in self.parameters_iter:
            if not p.requires_grad:
                continue
            # Use parameter's data properties (grad may be None before backward)
            p_device = p.device
            p_dtype = p.dtype
            p_numel = p.numel()
            p_bytes = p.element_size() * p_numel

            # Start new bucket if device/dtype changes or size would overflow
            if (
                not current_params
                or current_device != p_device
                or current_dtype != p_dtype
                or (current_size_bytes + p_bytes) > self.bucket_size_bytes
            ):
                flush_bucket()
                current_device = p_device
                current_dtype = p_dtype

            start = sum(n for _, n in current_offsets)
            current_params.append(p)
            current_offsets.append((start, p_numel))
            current_size_bytes += p_bytes

        # Flush the last bucket
        flush_bucket()

    def _register_param_hooks(self) -> None:
        """Register hooks that fire after grad is accumulated into p.grad.
        When a bucket's all params are ready, launch a single async all-reduce
        on the bucket's flat buffer.
        """
        self._hooks = []  # keep refs

        # Map parameter -> (bucket_index, offset_index)
        self._param_to_bucket: dict[int, Tuple[int, int]] = {}
        for b_idx, bucket in enumerate(self._buckets):
            for o_idx, p in enumerate(bucket.params):
                self._param_to_bucket[id(p)] = (b_idx, o_idx)

        for b_idx, bucket in enumerate(self._buckets):
            for o_idx, p in enumerate(bucket.params):
                def make_post_hook(param: nn.Parameter, bucket_index: int, offset_index: int):
                    def _post_hook(par: nn.Parameter):
                        # Called after grad has been accumulated into par.grad
                        if par.grad is None:
                            return
                        b = self._buckets[bucket_index]
                        start, numel = b.offsets[offset_index]
                        # Flatten and copy into bucket buffer
                        b.flat_buffer.narrow(0, start, numel).copy_(par.grad.view(-1))
                        b.ready_count += 1
                        # If all grads in bucket are ready, launch async all-reduce
                        if b.ready_count == len(b.params) and b.work is None:
                            b.work = dist.all_reduce(b.flat_buffer, op=dist.ReduceOp.SUM, async_op=True)
                    return _post_hook
                self._hooks.append(p.register_post_accumulate_grad_hook(make_post_hook(p, b_idx, o_idx)))

    @torch.no_grad()
    def on_train_batch_start(self) -> None:
        """Reset bucket state at the start of a new training step."""
        # Cancel previous state
        for b in self._buckets:
            b.ready_count = 0
            b.work = None
        # No need to clear flat buffers; they will be overwritten by next backward

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """
        Wait for all asynchronous bucket communications to finish, average grads,
        and scatter reduced gradients back into per-parameter .grad tensors.
        Call after loss.backward() and before optimizer.step().
        """
        if not self._buckets:
            return
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        t0 = time()
        # Wait for outstanding comms
        for b in self._buckets:
            if b.work is not None:
                b.work.wait()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self._time_waiting_for_comm += time() - t0

        # Average and scatter back
        inv_ws = 1.0 / float(world_size)
        for b in self._buckets:
            if b.ready_count == 0:
                continue  # nothing to do
            # Average in-place
            b.flat_buffer.mul_(inv_ws)
            # Scatter back to params' grad
            for (start, numel), p in zip(b.offsets, b.params):
                if p.grad is not None:
                    p.grad.view(-1)[:numel].copy_(b.flat_buffer.narrow(0, start, numel))
            # Reset for next iteration
            b.ready_count = 0
            b.work = None

