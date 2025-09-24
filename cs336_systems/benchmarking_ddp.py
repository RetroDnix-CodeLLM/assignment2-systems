import yaml, os
from argparse import ArgumentParser
from copy import deepcopy
from time import time

from cs336_basics import TransformerLM
from cs336_systems.naive_ddp import DDPIndividualParameters
from cs336_systems.minimal_ddp import DDPIndividualParametersMinimal
from cs336_systems.ddp_overlap_individual_parameters import DDPIndividualParametersOnAfterBackward
from cs336_systems.ddp_bucketed import DDPBucketed

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

def _setup_process_group(rank: int, world_size: int, backend: str = "gloo", device: str = "cpu") -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA backend requested but not available"
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device


def _cleanup_process_group():
    dist.barrier()
    dist.destroy_process_group()


DDP_Implement = None
def _ddp_worker(rank: int, world_size: int, ddp_impl, model: nn.Module, config: dict, backend: str = "gloo", device:str = "cpu", seed: int = 1234):
    device = _setup_process_group(rank, world_size, backend=backend, device=device)

    # Ensure different initializations per rank pre-broadcast
    torch.manual_seed(seed + rank)
    # Build baseline and ddp models
    non_parallel_model = model(**config["model"]).to(device)
    ddp_model = ddp_impl(deepcopy(non_parallel_model)).to(device)

    # Data: total batch is bs_per_rank * world_size; each rank sees a shard
    torch.manual_seed(seed)
    iters = config["data"]["iters"]
    bs_per_rank = config["data"]["batch_size"]
    total_bs = bs_per_rank * world_size

    x_all = np.random.randint(0, config["model"]["vocab_size"], size=(total_bs, config["model"]["context_length"])) 
    y_all = np.random.randint(0, config["model"]["vocab_size"], size=(total_bs, config["model"]["context_length"]))
    x_all = torch.tensor(x_all, dtype=torch.long, device=device)
    y_all = torch.tensor(y_all, dtype=torch.long, device=device)

    loss_fn = nn.CrossEntropyLoss()
    opt_ddp = optim.SGD(ddp_model.parameters(), lr=0.1)

    if rank == 0:
        t0 = time()
    for i in range(iters):
        opt_ddp.zero_grad(set_to_none=True)
        start = rank * bs_per_rank
        end = start + bs_per_rank
        out_ddp = ddp_model(x_all[start:end])
        loss_ddp = loss_fn(out_ddp.view(-1, out_ddp.size(-1)), y_all[start:end].view(-1))
        loss_ddp.backward()
        ddp_model.finish_gradient_synchronization()
        opt_ddp.step()

        # Shuffle for next iteration (deterministically across ranks)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100 + i)
            perm = torch.randperm(total_bs, device=device)
        x_all = x_all[perm]
        y_all = y_all[perm]
    if rank == 0:
        torch.cuda.synchronize()
        elapsed = time() - t0
        print(f"Rank {rank}: Completed {iters} iterations in {elapsed:.4f}s")
        print(f"Rank {rank}: Time waiting for communication: {ddp_model._time_waiting_for_comm:.4f}s")

    # Final cross-rank equivalence check for ddp model state
    # Gather each tensor in state dict to ensure all ranks agree
    for t in ddp_model.module.state_dict().values():
        gather_list = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(gather_list, t)
        for other in gather_list:
            assert torch.allclose(other, t), "Model state not synchronized across ranks"

    if rank == 0:
        print("Naive DDP demo completed successfully.")

    _cleanup_process_group()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes to launch")
    parser.add_argument("--model", type=str, default="xl", choices=["small", "medium", "large", "xl", "2.7B"], help="Model size to benchmark")
    parser.add_argument("--ddp_impl", type=str, default="naive", choices=["naive", "minimal", "overlap", "bucketed"], help="DDP implementation to use")
    args = parser.parse_args()

    world_size = args.world_size
    
    match args.ddp_impl:
        case "naive":
            DDP_Implement = DDPIndividualParameters
        case "minimal":
            DDP_Implement = DDPIndividualParametersMinimal
        case "overlap":
            DDP_Implement = DDPIndividualParametersOnAfterBackward
        case "bucketed":
            DDP_Implement = DDPBucketed
        case _:
            raise NotImplementedError(f"Unknown DDP implementation: {args.ddp_impl}")

    config = yaml.safe_load(open(f"config/{args.model}.yaml", "r"))
    config["model"].update({
        "vocab_size": 10000,
        "context_length": 512,
        "rope_theta": 100000,
        "device": "cuda"
    })

    # 使用 torch.multiprocessing 启动多个进程
    mp.spawn(_ddp_worker, args=(world_size, DDP_Implement, TransformerLM, config, "nccl", "cuda", 10), nprocs=world_size, join=True)
