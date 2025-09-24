from __future__ import annotations

import math
from typing import Any, Tuple
from jaxtyping import Float
import math
from einops import einsum
import torch
from torch import Tensor
import triton
import triton.language as tl

class FlashAttnV2Torch(torch.autograd.Function):
	"""
	Pure-PyTorch (no Triton) reference forward for FlashAttention-2.

	Notes
	- Implements the numerically-stable blockwise softmax accumulation
	  (equations analogous to FA2 Eq. 4-6 and 12):
		• Maintain running max m_i and partition l_i per query row.
		• Accumulate numerator acc_i = sum_k exp(S_ik) V_k in a blockwise manner.
		• At the end, O_i = acc_i / l_i and L_i = log(l_i) + m_i.
	- is_causal is accepted for API parity but ignored per assignment instructions.
	- We tile queries and keys with a minimum tile size of 16.
	- Backward is intentionally unimplemented here (raise NotImplementedError).
	"""

	@staticmethod
	def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:  # noqa: D401
		# Shapes: Q: [B, Nq, D], K: [B, Nk, D], V: [B, Nk, D]
		assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "Q, K, V must be 3D tensors"
		B, Nq, D = Q.shape
		Bk, Nk, Dk = K.shape
		Bv, Nv, Dv = V.shape
		assert B == Bk == Bv, "Batch size mismatch among Q, K, V"
		assert D == Dk == Dv, "Embedding dimension mismatch among Q, K, V"
		assert Nk == Nv, "Number of keys must match between K and V"

		dtype = Q.dtype
		device = Q.device
		scale = 1.0 / math.sqrt(D)

		# Tile sizes (at least 16 x 16 as per requirement)
		TQ = 16
		TK = 16

		# Outputs
		O = torch.empty((B, Nq, D), dtype=dtype, device=device)
		# logsumexp values per query row
		L = torch.empty((B, Nq), dtype=dtype, device=device)

		# Iterate over batches and tiles
		for b in range(B):
			Qb = Q[b]
			Kb = K[b]
			Vb = V[b]

			# Process query tiles
			for qs in range(0, Nq, TQ):
				qe = qs + TQ
				Qi = Qb[qs:qe, :]  # [TQ, D]

				# Initialize running values for this query tile
				# m_i: running max per query row; start with very small numbers
				m_i = torch.full((TQ,), -float("inf"), dtype=dtype, device=device)
				# l_i: running partition function per query row
				l_i = torch.zeros((TQ,), dtype=dtype, device=device)
				# acc_i: running numerator for output (sum exp(S) V)
				acc_i = torch.zeros((TQ, D), dtype=dtype, device=device)

				# Iterate over key tiles
				for ks in range(0, Nk, TK):
					ke = ks + TK
					Kj = Kb[ks:ke, :]  # [TK, D]
					Vj = Vb[ks:ke, :]  # [TK, D]

					# Scores for the current blocks: [TQ, TK]
					Sij = (Qi @ Kj.T) * scale

					# Update running max per row
					mij = torch.max(Sij, dim=1).values  # [TQ]
					new_m = torch.maximum(m_i, mij)  # [TQ]

					# Compute exp factors relative to new_m for stability
					exp_m_prev = torch.exp(m_i - new_m)  # [TQ]
					exp_S = torch.exp(Sij - new_m[:, None])  # [TQ, TK]

					# Update l_i and accumulator
					l_i = l_i * exp_m_prev + torch.sum(exp_S, dim=1)  # [TQ]
					acc_i = acc_i * exp_m_prev[:, None] + exp_S @ Vj  # [TQ, D]

					# Commit new max
					m_i = new_m

				# Finalize outputs for this query block
				O_block = acc_i / l_i[:, None]  # [TQ, D]
				L_block = torch.log(l_i) + m_i  # [TQ]

				O[b, qs:qe, :] = O_block
				L[b, qs:qe] = L_block

		# Save tensors needed for backward (L is required by tests to be present)
		ctx.save_for_backward(L, Q, K, V, O)
		# Store any non-tensor args on ctx if needed later (ignored for now)
		ctx.is_causal = bool(is_causal)

		return O

	@staticmethod
	def backward(ctx, dO: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
		"""Pure PyTorch backward using saved tensors.

		Computes gradients w.r.t Q, K, V following the standard attention
		backward with the FlashAttention trick of using the saved L (logsumexp)
		to reconstruct probabilities P = exp(S - L).

		dV = P^T @ dO
		dP_ik = dO_i · V_k
		D_i = sum_k P_ik * dP_ik = dO_i · O_i
		dS = P ⊙ (dP - D)
		dQ = dS @ K * scale
		dK = dS^T @ Q * scale
		"""
		(L, Q, K, V, O) = ctx.saved_tensors

		def _backward_impl(Q, K, V, O, dO, L):
			B, Nq, D = Q.shape
			scale = 1.0 / math.sqrt(D)
			# Scores and probabilities
			S = Q @ K.transpose(-1, -2) * scale  # [B, Nq, Nk]
			P = torch.exp(S - L.unsqueeze(-1))   # softmax via saved L

			# Grad w.r.t V
			dV = torch.einsum('bqk,bqd->bkd', P, dO)

			# Intermediate for dS
			dP = torch.einsum('bqd,bkd->bqk', dO, V)  # [B, Nq, Nk]
			D_vec = (dO * O).sum(dim=-1, keepdim=True)  # [B, Nq, 1]
			dS = P * (dP - D_vec)  # [B, Nq, Nk]

			# Grads for Q, K
			dQ = torch.einsum('bqk,bkd->bqd', dS, K) * scale
			dK = torch.einsum('bqk,bqd->bkd', dS, Q) * scale
			return dQ, dK, dV

		# Try to leverage torch.compile for performance; fallback if unavailable
		_compiled = None
		if hasattr(torch, 'compile'):
			try:
				_compiled = torch.compile(_backward_impl, fullgraph=False)
			except Exception:
				_compiled = None

		if _compiled is not None:
			dQ, dK, dV = _compiled(Q, K, V, O, dO, L)
		else:
			dQ, dK, dV = _backward_impl(Q, K, V, O, dO, L)

		# Return grads for inputs of forward: (Q, K, V, is_causal)
		return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr,K_ptr,V_ptr,O_ptr,L_ptr,
    stride_qb,stride_qq,stride_qd,
    stride_kb,stride_kk,stride_kd,
    stride_vb,stride_vk,stride_vd,
    stride_ob,stride_oq,stride_od,
    stride_lb,stride_lq,
    N_queries,
    N_keys,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_queries, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_keys, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_keys, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_queries, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_queries,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Tk = tl.cdiv(N_keys, K_TILE_SIZE)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        prev_mi = mi
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        if is_causal:
            q_index = (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))[:, None]
            k_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
            mask = tl.where(k_index <= q_index, tl.zeros((Q_TILE_SIZE, K_TILE_SIZE),dtype=tl.float32), tl.full((Q_TILE_SIZE, K_TILE_SIZE), -1e6, dtype=tl.float32))
            Sij = Sij + mask
        mi = tl.maximum(prev_mi, tl.max(Sij, axis=-1))
        Pij = tl.exp(Sij - mi[:, None])
        scaling_factor = tl.exp(prev_mi - mi)
        li = scaling_factor * li + tl.sum(Pij, axis=-1)
        Oi = scaling_factor[:, None] * Oi + tl.dot(Pij.to(Vj.dtype), Vj)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    Oi = (1 / li)[:, None] * Oi
    li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, li, boundary_check=(0,))


class FlashAttnV2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "B queries d_k"],
        K: Float[Tensor, "B keys d_k"],
        V: Float[Tensor, "B keys d_v"],
        is_causal: bool = False,
    ):
        ctx.save_for_backward(Q, K, V)
        Bq = 16
        Bk = 16
        D = Q.shape[2]
        B = Q.shape[0]
        N_queries = Q.shape[1]
        N_keys = K.shape[1]
        Tq = triton.cdiv(N_queries, Bq)
        O = torch.empty((B, N_queries, D), device=Q.device)
        L = torch.empty((B, N_queries), device=Q.device)
        scale = 1 / math.sqrt(D)
        grid = (Tq, B)
        flash_fwd_kernel[grid](
            Q,K,V,O,L,
            Q.stride(0),Q.stride(1),Q.stride(2),
            K.stride(0),K.stride(1),K.stride(2),
            V.stride(0),V.stride(1),V.stride(2),
            O.stride(0),O.stride(1),O.stride(2),
            L.stride(0),L.stride(1),
            N_queries,N_keys,
            scale,D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )
        ctx.save_for_backward(L, O)
        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

__all__ = ["FlashAttnV2Torch", "FlashAttnV2Triton"]  # type: ignore