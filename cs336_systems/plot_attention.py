from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
	# Resolve project root and I/O paths
	root = Path(__file__).resolve().parent.parent
	results_dir = root / "results"
	csv_path = results_dir / "benchmarking_attention_triton.csv"
	out_path = results_dir / "attention_performance_d128.png"

	if not csv_path.exists():
		print(f"[ERROR] Results CSV not found: {csv_path}")
		return 1

	# Load data
	df = pd.read_csv(csv_path)

	# Filter for d_model == 128
	df = df[df["d_model"] == 128].copy()
	if df.empty:
		print("[ERROR] No rows with d_model == 128 found in the CSV.")
		return 1

	# Ensure expected columns exist
	required_cols = {"method_name", "d_model", "seq_len", "forward_time(s)", "mem(MB)"}
	missing = required_cols - set(df.columns)
	if missing:
		print(f"[ERROR] Missing required columns in CSV: {sorted(missing)}")
		return 1

	# Prepare x-axis as powers of two (k = log2(seq_len))
	# Keep a stable ordering of sequence lengths
	seq_lens = sorted(df["seq_len"].unique().tolist())
	ks = [int(np.log2(s)) for s in seq_lens]

	# Methods order for consistent plotting
	methods = [
		"Naive Attention",
		"PyTorch Built-in",
		"FlashAttention v2 Triton",
	]
	# In case CSV contains a subset or different order
	present_methods = [m for m in methods if m in df["method_name"].unique()]
	# Assign consistent colors
	color_map = {
		"Naive Attention": "#1f77b4",  # blue
		"PyTorch Built-in": "#ff7f0e",  # orange
		"FlashAttention v2 Triton": "#2ca02c",  # green
	}

	# Build a pivot-friendly dict by method for fast lookup
	# For each method, build arrays of forward_time and mem aligned to seq_lens order
	fwd_by_method: dict[str, list[float]] = {}
	mem_by_method: dict[str, list[float]] = {}
	for m in present_methods:
		sub = df[df["method_name"] == m]
		# Map seq_len -> metric
		fwd_map = {int(r.seq_len): float(r["forward_time(s)"]) for _, r in sub.iterrows()}
		mem_map = {int(r.seq_len): float(r["mem(MB)"]) for _, r in sub.iterrows()}
		fwd_by_method[m] = [fwd_map.get(s, np.nan) for s in seq_lens]
		mem_by_method[m] = [mem_map.get(s, np.nan) for s in seq_lens]

	# Create figure with twin y-axes
	plt.figure(figsize=(10, 6))
	ax_time = plt.gca()
	ax_mem = ax_time.twinx()

	x = np.arange(len(seq_lens), dtype=float)  # positions 0..N-1

	# Plot forward time lines on left axis
	for m in present_methods:
		ax_time.plot(
			x,
			fwd_by_method[m],
			marker="o",
			linewidth=2,
			label=m,
			color=color_map.get(m, None),
		)

	ax_time.set_ylabel("Forward Time (s)")
	ax_time.set_yscale('log', base=2)
	ax_time.grid(True, axis="y", linestyle=":", alpha=0.5)

	# Plot grouped bars for memory on right axis
	n_methods = len(present_methods)
	total_bar_width = 0.75  # portion of the x-slot used by bars
	bar_width = total_bar_width / max(n_methods, 1)
	# Center the groups around each x position
	offsets = (
		np.linspace(-total_bar_width / 2 + bar_width / 2, total_bar_width / 2 - bar_width / 2, n_methods)
		if n_methods > 0
		else np.array([0.0])
	)

	for idx, m in enumerate(present_methods):
		ax_mem.bar(
			x + offsets[idx],
			mem_by_method[m],
			width=bar_width,
			label=f"{m} (Mem)",
			color=color_map.get(m, None),
			alpha=0.35,
			edgecolor="none",
		)

	ax_mem.set_ylabel("Memory (MB)")
	ax_mem.set_yscale('log', base=2)
	ax_mem.legend(loc="upper left", bbox_to_anchor=(0, 0.85), frameon=True)

	# X-axis ticks and labels as powers of 2
	# Labels like 2^8, 2^9, ... matching seq_lens
	ax_time.set_xticks(x)
	ax_time.set_xticklabels([f"2^{k}" for k in ks])
	ax_time.set_xlabel("Sequence Length (2^k)")

	# Build combined legend: use one legend for lines, another for bars, or merge handles
	line_handles, line_labels = ax_time.get_legend_handles_labels()
	bar_handles, bar_labels = ax_mem.get_legend_handles_labels()

	# Prefer a single legend showing only method once; the color encodes both line and bar.
	# We'll show method names once (lines) and add a small note for bars in the title.
	if line_handles:
		leg = ax_time.legend(line_handles, line_labels, loc="upper left", frameon=True)
		for legobj in leg.legend_handles:
			legobj.set_linewidth(3.0)

	# Title
	ax_time.set_title("Attention Performance at d_model = 128\nLines: Forward Time | Bars: Memory")

	plt.tight_layout()

	# Ensure output directory exists and save
	results_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=150)
	plt.close()

	print(f"[INFO] Figure saved to: {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

