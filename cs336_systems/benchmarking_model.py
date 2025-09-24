import yaml, argparse
import numpy as np
from time import time
from tqdm import trange

import torch
import pandas as pd

from typing import Dict

from cs336_basics import TransformerLM, cross_entropy_loss

def BenchMarkingModel(config: Dict, repeat_times: int, run_backward: bool) -> float:
    print("Using belowing model config:\n", config)

    # 初始化模型
    model = TransformerLM(**config["model"])

    dataset = np.random.randint(0, config["model"]["vocab_size"], size=(config["data"]["batch_size"], config["model"]["context_length"])) 
    targets = np.random.randint(0, config["model"]["vocab_size"], size=(config["data"]["batch_size"], config["model"]["context_length"] ))
    dataset = torch.tensor(dataset, dtype=torch.long, device=config["model"]["device"])
    targets = torch.tensor(targets, dtype=torch.long, device=config["model"]["device"])

    # 预热
    for _ in trange(5, desc="Warming up..."):
        outputs = model(dataset)
        # loss = cross_entropy_loss(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        # loss.backward()

    # 测试5次取平均
    t0 = time()
    for _ in trange(repeat_times, desc="Benchmarking..."):
        outputs = model(dataset)
        if run_backward:
            loss = cross_entropy_loss(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
    t1 = time()

    del model
    torch.cuda.empty_cache()
    return (t1 - t0) / repeat_times

if __name__ == "__main__":
    configs = ["small", "medium", "large", "xl", "2.7B"]
    # configs = ["small"]
    result = pd.DataFrame(columns=["Model", "Forward Time (s)", "Forward + Backward Time (s)"])
    for config_name in configs:

        config = yaml.safe_load(open(f"config/{config_name}.yaml", "r"))
        config["model"].update({
            "vocab_size": 10000,
            "context_length": 512,
            "rope_theta": 100000,
            "device": "cuda"
        })

        avg_t1 = BenchMarkingModel(config, 5, run_backward=False)
        print("Forward Time:", avg_t1)
        avg_t2 = BenchMarkingModel(config, 5, run_backward=True)
        print("Forward + Backward Time:", avg_t2)

        result = pd.concat([result, pd.DataFrame([[config_name, f"{avg_t1:.4f}", f"{avg_t2:.4f}"]], columns=result.columns)], ignore_index=True)
        result.to_csv("./results/benchmarking_results.csv", index=False)