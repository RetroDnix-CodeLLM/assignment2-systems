from __future__ import annotations

from typing import Iterable
from time import time

import torch
import torch.distributed as dist
import torch.nn as nn

class DDPIndividualParametersOnAfterBackward(nn.Module):
    """
    DDP wrapper that overlaps gradient communication with the backward pass
    by launching asynchronous all-reduces as soon as a parameter's gradient
    is produced by autograd (via per-parameter autograd hooks).

    Flow:
    - On init: broadcast parameters and buffers from rank 0 to all ranks.
    - During backward: hooks on parameters kick off async all_reduce on grads.
    - After backward, before optimizer.step(): call finish_gradient_synchronization()
      to wait for all outstanding communications and average grads by world size.

    Notes:
    - Parameters are deduplicated to handle tied/shared weights.
    - Non-grad parameters are still broadcasted on init to keep states in sync.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized. Call dist.init_process_group first.")
        self.module = module
        self._dedup_param_ids = set()
        self._time_waiting_for_comm = 0.0
        # Store async works and corresponding params to scale after wait
        self._grad_works: list[tuple[nn.Parameter, dist.Work]] = []
        # Register autograd hooks for overlapping communication
        self._register_param_hooks()
        # Ensure all ranks start from rank 0's parameters and buffers
        self._broadcast_model_state_from_rank0()

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
        for p in self.parameters_iter:
            # Broadcast parameter data (including those with requires_grad=False)
            dist.broadcast(p.data, src=0)
        # Also broadcast buffers to keep running stats etc. in sync
        for b in self.module.buffers(recurse=True):
            dist.broadcast(b.data, src=0)

    def _register_param_hooks(self) -> None:
        """Register hooks that fire after grad is accumulated into p.grad.
        This launches async all-reduce on the accumulated gradient to overlap
        communication with remaining backward computation.
        """
        # Keep a reference to hooks to avoid garbage collection
        self._hooks = []
        # Clear any pending works from a previous step (safety)
        self._grad_works.clear()

        for p in self.parameters_iter:
            if not p.requires_grad:
                continue

            def make_post_hook(param: nn.Parameter):
                def _post_hook(p: nn.Parameter):
                    # Called after gradient has been accumulated into p.grad
                    if p.grad is None:
                        return
                    # Launch async all-reduce in-place on the accumulated grad
                    work = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                    # Record the work handle and the parameter for later scaling.
                    self._grad_works.append((param, work))
                return _post_hook

            # Use post-accumulate grad hook to ensure we see p.grad after accumulation
            self._hooks.append(p.register_post_accumulate_grad_hook(make_post_hook(p)))

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """
        Wait for all asynchronous grad communications to finish and average grads.
        Call after loss.backward() and before optimizer.step().
        """
        if not self._grad_works:
            return
        world_size = dist.get_world_size()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time()
        # Wait for all communications to finish
        for param, work in self._grad_works:
            work.wait()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._time_waiting_for_comm += time() - t0
        # Scale grads by world size
        for param, _ in self._grad_works:
            if param.grad is not None:
                param.grad /= world_size
        # Clear for next iteration
        self._grad_works.clear()