from __future__ import annotations

from typing import Iterable, Optional
from time import time

import torch
import torch.distributed as dist
import torch.nn as nn


class DDPIndividualParametersMinimal(nn.Module):
	"""
	A minimal DDP wrapper that:
	- Broadcasts parameters and buffers from rank 0 to all ranks on init
	- After backward, communicates a single flattened gradient tensor across ranks (one all_reduce) and averages by world size
	"""

	def __init__(self, module: nn.Module):
		super().__init__()
		if not dist.is_initialized():
			raise RuntimeError("Process group not initialized. Call dist.init_process_group first.")
		self.module = module
		self._dedup_param_ids = set()

		self._time_waiting_for_comm = 0.0  # Optional: track time spent waiting for communication
		# Cached structures for flattened gradient communication
		self._flat_grad_buffer: Optional[torch.Tensor] = None
		self._flat_param_views: list[nn.Parameter] = []
		self._flat_slices: list[tuple[int, int]] = []  # (start, end) per param in flat buffer

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

	def _ensure_flatten_plan(self) -> None:
		"""
		Build and cache the mapping from parameters to a single flat gradient buffer.
		Assumes parameter set (and order) is static after construction.
		"""
		if self._flat_grad_buffer is not None:
			return
		# Collect unique, trainable parameters in a stable order
		params: list[nn.Parameter] = [p for p in self.parameters_iter if p.requires_grad]
		if not params:
			# Nothing to sync; create a trivial buffer
			self._flat_grad_buffer = torch.tensor([], device=next(self.module.parameters()).device)
			self._flat_param_views = []
			self._flat_slices = []
			return
		device = params[0].device
		dtype = params[0].dtype
		# Sanity: all params should share device & dtype in this simple impl
		for p in params:
			if p.device != device:
				raise RuntimeError("All parameters must be on the same device for flattening.")
			if p.dtype != dtype:
				raise RuntimeError("All parameters must share the same dtype for flattening.")
			total_numel = sum(p.numel() for p in params)
		self._flat_grad_buffer = torch.zeros(total_numel, device=device, dtype=dtype)
		self._flat_param_views = params
		# Compute slices (start, end) per param
		self._flat_slices = []
		cursor = 0
		for p in params:
			start = cursor
			end = start + p.numel()
			self._flat_slices.append((start, end))
			cursor = end

	@torch.no_grad()
	def finish_gradient_synchronization(self) -> None:
		"""
		Call after loss.backward() and before optimizer.step().
		Flattens all gradients into a single contiguous tensor, performs one all_reduce,
		then scatters the averaged gradients back to the individual parameter.grad tensors.
		"""
		self._ensure_flatten_plan()
		world_size = dist.get_world_size()
		if self._flat_grad_buffer is None or self._flat_grad_buffer.numel() == 0:
			return  # nothing to sync
		# Pack gradients into flat buffer (zeros for unused grads to keep shapes consistent)
		for p, (start, end) in zip(self._flat_param_views, self._flat_slices):
			view = self._flat_grad_buffer[start:end]
			if p.grad is None:
				view.zero_()
			else:
				view.copy_(p.grad.view(-1))
		# One collective communication for all grads
		torch.cuda.synchronize()
		t0 = time()
		dist.all_reduce(self._flat_grad_buffer, op=dist.ReduceOp.SUM)
		torch.cuda.synchronize()
		self._time_waiting_for_comm += time() - t0
		self._flat_grad_buffer /= world_size
		# Scatter averaged gradients back (preserve None for unused params)
		for p, (start, end) in zip(self._flat_param_views, self._flat_slices):
			if p.grad is None:
				continue
			p.grad.view(-1).copy_(self._flat_grad_buffer[start:end])
