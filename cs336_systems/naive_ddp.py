from __future__ import annotations

import os
from copy import deepcopy
from typing import Iterable, Optional
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


class DDPIndividualParameters(nn.Module):
	"""
	A minimal DDP wrapper that:
	- Broadcasts parameters and buffers from rank 0 to all ranks on init
	- After backward, all-reduces each parameter's gradient individually and averages by world size

	This is intentionally naive and meant for educational purposes.
	"""

	def __init__(self, module: nn.Module):
		super().__init__()
		if not dist.is_initialized():
			raise RuntimeError("Process group not initialized. Call dist.init_process_group first.")
		self.module = module
		self._dedup_param_ids = set()

		self._time_waiting_for_comm = 0.0  # Optional: track time spent waiting for communication

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

	@torch.no_grad()
	def finish_gradient_synchronization(self) -> None:
		"""
		Call after loss.backward() and before optimizer.step().
		All-reduces each parameter's gradient across ranks and averages.
		"""
		world_size = dist.get_world_size()
		for p in self.parameters_iter:
			if not p.requires_grad:
				continue
			if p.grad is None:
				# Parameter not used in this backward pass
				continue
			torch.cuda.synchronize()
			t0 = time()
			dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
			torch.cuda.synchronize()
			self._time_waiting_for_comm += time() - t0
			p.grad /= world_size


