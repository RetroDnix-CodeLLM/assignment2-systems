import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch import Tensor

import math
import jaxtyping

from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

from cs336_basics.model_utils import softmax

def cross_entropy_loss(
    logits: Float[Tensor, "... vocab_size"],
    targets: Float[Tensor, "..."],
) -> Float[Tensor, ""]:
    """
    Compute the cross-entropy loss between logits and integer targets.
    
    Args:
        logits: Tensor of shape (..., vocab_size), raw output scores (unnormalized).
        targets: Tensor of shape (...), with integer class indices.
    
    Returns:
        Scalar tensor with the mean cross-entropy loss over all entries.
    """
    log_probs = F.log_softmax(logits, dim=-1)   # shape: (..., vocab_size)
    
    # Gather log-probabilities corresponding to targets
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # shape: (...)
    
    loss = -target_log_probs  # negative log-likelihood
    return loss.mean()        # mean over all entries

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        # Validate input parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        # Initialize optimizer with parameter groups
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Perform stepweight decay
                p.mul_(1 - lr * weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute and apply learning rate
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def get_cosine_annealing_lr(    
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
    """
    Compute the learning rate using cosine annealing.
    
    Args:
        t: Current time step.
        aMax: Maximum learning rate.
        aMin: Minimum learning rate.
        Tw: Warmup period (in steps).
        Tc: Total cycle period (in steps).
    
    Returns:
        Computed learning rate at time step t.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(torch.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        return min_learning_rate

class lr_scheduler:
    def __init__(
            self, 
            optimizer: Optimizer,
            max_learning_rate: float,
            min_learning_rate: float,
            warmup_iters: int,
            cosine_cycle_iters: int
        ):
        """
        Initialize the learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule.
            max_learning_rate: Maximum learning rate.
            min_learning_rate: Minimum learning rate.
            warmup_iters: Number of warmup iterations.
            cosine_cycle_iters: Total cycle period for cosine annealing.
        """
        self.optimizer = optimizer
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        self.last_epoch = 0

    def step(self):
        """
        Update the learning rate for each parameter group in the optimizer.
        """
        self.last_epoch += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = get_cosine_annealing_lr(self.last_epoch, self.max_learning_rate, self.min_learning_rate, self.warmup_iters, self.cosine_cycle_iters)
    
    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """
    Clip gradients of parameters to a maximum L2 norm.
    
    Args:
        parameters: Iterable of model parameters.
        max_l2_norm: Maximum allowed L2 norm for gradients.
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)