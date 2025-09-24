import yaml, argparse
import numpy as np

import torch

from typing import Dict

from cs336_basics import TransformerLM, cross_entropy_loss, AdamW

def BenchMarkingModelMemory(config: Dict, repeat_times: int, run_backward: bool) -> float:
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    print("Using belowing model config:\n", config)

    # 初始化模型
    model = TransformerLM(**config["model"])
    optimizer = AdamW(model.parameters(), lr=0.005)

    dataset = np.random.randint(0, config["model"]["vocab_size"], size=(config["data"]["batch_size"], config["model"]["context_length"]))
    target = np.random.randint(0, config["model"]["vocab_size"], size=(config["data"]["batch_size"], config["model"]["context_length"] ))
    target = torch.tensor(target, dtype=torch.long, device=config["model"]["device"])
    print("Input dataset shape:", dataset.shape)

    for step in range(repeat_times):
        optimizer.zero_grad()
        logits = model(dataset)

        if run_backward:
            loss = cross_entropy_loss(logits, target)
            loss.backward()
            optimizer.step()
    
    length = config["model"]["context_length"]
    torch.cuda.memory._dump_snapshot(f"memory_snapshot_{length}_backward_{run_backward}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == "__main__":
    for length in [128, 256, 512]:
        config = yaml.safe_load(open(f"config/2.7B.yaml", "r"))
        config["model"].update({
            "vocab_size": 10000,
            "context_length": length,
            "rope_theta": 100000,
            "device": "cuda"
        })

        BenchMarkingModelMemory(config, repeat_times=10, run_backward=True)
