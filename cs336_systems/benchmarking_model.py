import yaml, argparse
import numpy as np
import timeit

from typing import Dict

from cs336_basics import BasicsTransformerLM

def BenchMarkingModel(config: Dict) -> float:
    config = yaml.safe_load(open(config, "r"))
    print("Using belowing model config:\n", config)

    # 初始化模型
    model = BasicsTransformerLM(**config["model"])

    dataset = np.random.randint(0, config["model"]["vocab_size"], size=(config["data"]["batch_size"], config["data"]["seq_len"]))
    print("Input dataset shape:", dataset.shape)

    # 预热
    for _ in range(2):
        _ = model(dataset)

    # 测试5次取平均
    times = timeit.repeat(lambda: model(dataset), number=1, repeat=5)
    avg_time = sum(times) / len(times)
    return avg_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking Model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Model config path")
    args = parser.parse_args()