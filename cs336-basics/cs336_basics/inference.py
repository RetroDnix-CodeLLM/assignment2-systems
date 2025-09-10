import yaml
import torch
from rich import print
from argparse import ArgumentParser

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.model_utils import softmax_with_temperature

default_ckpt_path = "./checkpoints/run_20250715_125311/ckpt_40000.pth"
default_model_config = "./config/train_tinystories.yaml"

if __name__ == "__main__":
    parser = ArgumentParser(description="Load a Transformer Language Model")
    parser.add_argument("--ckpt_path", type=str, default=default_ckpt_path, help="Path to the model checkpoint")
    parser.add_argument("--config", type=str, default=default_model_config, help="Path to the configuration file")
    parser.add_argument("--prompt", "-p", type=str, default="Once upon a time", help="Prompt to start the text generation")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for softmax scaling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling threshold")
    args = parser.parse_args()
    
    # Load the configuration
    print("[bold blue]Loading model configuration...[/bold blue]")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load the tokenizer
    print("[bold blue]Loading tokenizer...[/bold blue]")
    tokenizer = BPETokenizer(
        vocab = config["tokenizer"]["vocab_file"],
        merges = config["tokenizer"]["merges_file"],
        special_tokens=config["tokenizer"]["special_tokens"]
    )
    eos_id = tokenizer.vocab["<|endoftext|>".encode("utf-8")]

    # Load the model
    print("[bold blue]Loading model from checkpoint...[/bold blue]")
    ckpt = torch.load(args.ckpt_path)
    model = TransformerLM(**config["model"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config["training"]["device"])
    model.eval()

    input = tokenizer.encode(args.prompt)
    input = torch.tensor(input, dtype=torch.long).unsqueeze(0)  # Float[Tensor, "batch seq"]
    input = input.to(config["training"]["device"])

    print(f"[bold]{args.prompt}[/bold]", end="")
    output = []
    while (not output or output[-1] != eos_id) and len(output) < 256:
        with torch.no_grad():
            logits = model(input) # Float[Tensor, "batch seq vocab"]
            distribution = softmax_with_temperature(logits[:, -1, :], args.temperature)

            # 实现top-p采样
            sorted_probs, sorted_indices = torch.sort(distribution, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > args.top_p  # Top-p threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0  # Keep the first token
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            distribution[0, indices_to_remove] = 0  # Set low probability for removed

            next_token = torch.multinomial(distribution, num_samples=1)
            input = torch.cat([input, next_token], dim=1)

            next_token = next_token.item()
            output.append(next_token)
            next_char = tokenizer.decode(next_token)
            print(f"{next_char}", end="", flush=True)
    
    print("")
    