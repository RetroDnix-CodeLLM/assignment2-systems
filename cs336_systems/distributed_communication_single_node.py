import argparse
import os
import socket
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------
# Utilities
# ------------------------------


def find_free_port() -> int:
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(("127.0.0.1", 0))
	addr, port = s.getsockname()
	s.close()
	return port


def mib_to_numel_f32(mib: int) -> int:
	# MiB to bytes, then to float32 elements (4 bytes)
	return (mib * 1024 * 1024) // 4


def size_label(mib: int) -> str:
	return f"{mib}MiB" if mib < 1024 else f"{mib // 1024}GiB"


@dataclass
class BenchConfig:
	backend: str  # "gloo" or "nccl"
	device_type: str  # "cpu" or "cuda"
	world_size: int
	sizes_mib: List[int]
	warmup_iters: int
	iters_small: int
	iters_medium: int
	iters_large: int
	iters_xlarge: int
	output_csv: str


def choose_iters_for_size(mib: int, cfg: BenchConfig) -> int:
	# Heuristic to keep each setting under ~5 minutes overall.
	# Tighter for CPU/Gloo and very large sizes.
	if mib < 10:
		return cfg.iters_small
	if mib < 100:
		return cfg.iters_medium
	if mib < 1024:
		return cfg.iters_large
	return cfg.iters_xlarge


def effective_bandwidth_mib_per_s(mib: int, world_size: int, seconds: float) -> float:
	# Effective bandwidth for all-reduce (ring): 2 * (n-1)/n * message_size / time
	# Using MiB for size and seconds for time -> MiB/s
	if seconds <= 0:
		return float("inf")
	factor = 2.0 * (world_size - 1) / world_size
	return factor * mib / seconds


def init_process_group(rank: int, world_size: int, backend: str, port: int):
	dist.init_process_group(
		backend=backend,
		init_method=f"tcp://127.0.0.1:{port}",
		world_size=world_size,
		rank=rank,
		timeout=torch.distributed.constants.default_pg_timeout,
	)


def cleanup_process_group():
	if dist.is_initialized():
		dist.destroy_process_group()


def benchmark_worker(rank: int, cfg: BenchConfig, port: int, visible_cuda_count: int):
	use_cuda = cfg.device_type == "cuda"

	# Device setup for NCCL
	if use_cuda:
		assert cfg.backend == "nccl", "CUDA device requires NCCL backend"
		# Map rank->device index in the visible set [0..visible_cuda_count-1]
		device_index = rank % visible_cuda_count
		torch.cuda.set_device(device_index)
		device = torch.device(f"cuda:{device_index}")
	else:
		device = torch.device("cpu")

	# Initialize process group
	init_process_group(rank, cfg.world_size, cfg.backend, port)
	try:
		# Warmup: small tensor to establish communicators
		warm_tensor = torch.ones(8, device=device, dtype=torch.float32)
		for _ in range(2):
			dist.all_reduce(warm_tensor, op=dist.ReduceOp.SUM)
		dist.barrier()

		results = []  # Collect per-size metrics on rank 0

		for mib in cfg.sizes_mib:
			numel = mib_to_numel_f32(mib)
			# Allocate tensor
			try:
				x = torch.randn(numel, dtype=torch.float32, device=device)
			except RuntimeError as e:
				# Likely OOM; skip this size
				if rank == 0:
					print(f"[WARN] Skipping size {size_label(mib)} due to allocation error: {e}")
				dist.barrier()
				continue

			# Cache warmup to stabilize timings
			for _ in range(cfg.warmup_iters):
				dist.all_reduce(x, op=dist.ReduceOp.SUM)
				if use_cuda:
					torch.cuda.synchronize()
			dist.barrier()

			iters = choose_iters_for_size(mib, cfg)

			# Time multiple iters and compute average per-rank time
			if use_cuda:
				torch.cuda.synchronize()
			t0 = time.perf_counter()
			for _ in range(iters):
				dist.all_reduce(x, op=dist.ReduceOp.SUM)
			if use_cuda:
				torch.cuda.synchronize()
			t1 = time.perf_counter()
			avg_time = (t1 - t0) / iters

			# Reduce to get worst-case (max) average time across ranks
			t_tensor = torch.tensor([avg_time], device=device)
			dist.reduce(t_tensor, dst=0, op=dist.ReduceOp.MAX)

			if rank == 0:
				worst_avg_time = float(t_tensor.item())
				bw = effective_bandwidth_mib_per_s(mib, cfg.world_size, worst_avg_time)
				results.append(
					{
						"backend": cfg.backend,
						"device": cfg.device_type,
						"world_size": cfg.world_size,
						"size_mib": mib,
						"avg_time_s": worst_avg_time,
						"bandwidth_mib_s": bw,
						"iters": iters,
					}
				)

			# Make sure all ranks align before next size
			dist.barrier()

		# Rank 0 writes results to CSV (append)
		if rank == 0 and results:
			df = pd.DataFrame(results)
			os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)
			# Append or create
			header = not os.path.exists(cfg.output_csv)
			df.to_csv(cfg.output_csv, mode="a", header=header, index=False)

	finally:
		cleanup_process_group()


def run_bench_for_combo(backend: str, device_type: str, world_size: int, sizes_mib: List[int], csv_path: str):
	# Configure iteration counts based on backend/device for run time control
	if device_type == "cpu":
		warmup_iters = 2
		iters_small = 30   # < 10 MiB
		iters_medium = 10  # < 100 MiB
		iters_large = 3    # < 1 GiB
		iters_xlarge = 1   # >= 1 GiB
	else:  # cuda
		warmup_iters = 5
		iters_small = 50
		iters_medium = 20
		iters_large = 5
		iters_xlarge = 2

	cfg = BenchConfig(
		backend=backend,
		device_type=device_type,
		world_size=world_size,
		sizes_mib=sizes_mib,
		warmup_iters=warmup_iters,
		iters_small=iters_small,
		iters_medium=iters_medium,
		iters_large=iters_large,
		iters_xlarge=iters_xlarge,
		output_csv=csv_path,
	)

	# Setup NCCL environment when using CUDA
	visible_cuda_count = torch.cuda.device_count()
	if device_type == "cuda":
		if visible_cuda_count == 0:
			print("[WARN] No CUDA devices available, skipping NCCL benchmarks.")
			return
		if world_size > visible_cuda_count:
			print(
				f"[WARN] Requested world_size={world_size} exceeds available GPUs={visible_cuda_count}, skipping."
			)
			return

		# Limit visible devices to the first 'world_size' GPUs for this run
		os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
		visible_cuda_count = world_size

	# Find a free port for this run
	port = find_free_port()

	# Spawn processes
	mp.spawn(
		benchmark_worker,
		args=(cfg, port, visible_cuda_count),
		nprocs=world_size,
		join=True,
	)


def plot_results(csv_path: str, png_path: str):
	if not os.path.exists(csv_path):
		print(f"[WARN] No CSV found at {csv_path}, nothing to plot.")
		return
	df = pd.read_csv(csv_path)
	if df.empty:
		print("[WARN] CSV is empty, nothing to plot.")
		return

	# Plot 1: Time vs Size (log-log)
	plt.figure(figsize=(10, 6))
	for (backend, device, world_size), g in df.groupby(["backend", "device", "world_size"]):
		label = f"{backend.upper()}-{device} (p={world_size})"
		g_sorted = g.sort_values("size_mib")
		plt.plot(g_sorted["size_mib"], g_sorted["avg_time_s"] * 1000, marker="o", label=label)
	plt.xscale("log")
	plt.yscale("log")
	plt.xlabel("All-Reduce payload size (MiB)")
	plt.ylabel("Average time per all-reduce (ms)")
	plt.title("All-Reduce Runtime (Single Node, Multi-Process)")
	plt.grid(True, which="both", ls=":", alpha=0.5)
	plt.legend()
	os.makedirs(os.path.dirname(png_path), exist_ok=True)
	plt.tight_layout()
	plt.savefig(png_path)
	plt.close()

	# Plot 2: Bandwidth vs Size (semilogx)
	png_path_bw = os.path.splitext(png_path)[0] + "_bandwidth.png"
	plt.figure(figsize=(10, 6))
	for (backend, device, world_size), g in df.groupby(["backend", "device", "world_size"]):
		label = f"{backend.upper()}-{device} (p={world_size})"
		g_sorted = g.sort_values("size_mib")
		plt.semilogx(g_sorted["size_mib"], g_sorted["bandwidth_mib_s"] / 1024.0, marker="o", label=label)
	plt.xlabel("All-Reduce payload size (MiB)")
	plt.ylabel("Effective bandwidth (GiB/s)")
	plt.title("All-Reduce Effective Bandwidth (Single Node, Multi-Process)")
	plt.grid(True, which="both", ls=":", alpha=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(png_path_bw)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Benchmark all-reduce in single-node multi-process setup.")
	parser.add_argument(
		"--backends",
		type=str,
		default="gloo,nccl",
		help="Comma-separated backends to test: gloo (CPU), nccl (GPU).",
	)
	parser.add_argument(
		"--world-sizes",
		type=str,
		default="2,4,6",
		help="Comma-separated process counts to test.",
	)
	parser.add_argument(
		"--sizes-mb",
		type=str,
		default="1,10,100,1024",
		help="Comma-separated sizes in MiB for float32 payloads. Use 1024 for 1GiB.",
	)
	parser.add_argument(
		"--output-csv",
		type=str,
		default="results/benchmarking_allreduce.csv",
		help="CSV file to append results to.",
	)
	parser.add_argument(
		"--output-png",
		type=str,
		default="results/benchmarking_allreduce.png",
		help="PNG file to save plot to (a second _bandwidth.png will also be created).",
	)
	args = parser.parse_args()

	backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
	world_sizes = [int(x.strip()) for x in args.world_sizes.split(",") if x.strip()]
	sizes_mib = [int(x.strip()) for x in args.sizes_mb.split(",") if x.strip()]

	# Only allow the requested pairs from the prompt: Gloo+CPU, NCCL+GPU
	backend_device_pairs: List[Tuple[str, str]] = []
	if "gloo" in backends:
		backend_device_pairs.append(("gloo", "cpu"))
	if "nccl" in backends:
		backend_device_pairs.append(("nccl", "cuda"))

	# Run combinations sequentially
	for backend, device in backend_device_pairs:
		for ws in world_sizes:
			print(f"\n=== Running benchmark: backend={backend}, device={device}, world_size={ws} ===")
			run_bench_for_combo(backend, device, ws, sizes_mib, args.output_csv)

	# Plot results
	plot_results(args.output_csv, args.output_png)


if __name__ == "__main__":
	# Slightly reduce CPU thread over-subscription for Gloo runs
	try:
		torch.set_num_threads(max(1, os.cpu_count() // 2))
	except Exception:
		pass
	main()

