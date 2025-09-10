import wandb
import torch
import numpy as np
from rich import print
from rich.progress import Progress

import yaml
import os
from datetime import datetime
from os.path import join

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.optimize import AdamW, cross_entropy_loss, lr_scheduler, gradient_clipping
from cs336_basics.model import TransformerLM
from cs336_basics.data_loader import DataLoader
from cs336_basics.checkpoints import save_checkpoint, load_checkpoint

def train_loop(
        run_name: str, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: lr_scheduler, 
        dataloader: DataLoader, 
        max_steps: int,
        save_steps: int,
        save_path: str | os.PathLike
    ) -> None:
    """
    Train the model using the provided dataloader and optimizer.
    
    Args:
        model: The model to train.
        optimizer: The optimizer to use for training.
        dataloader: DataLoader instance providing batches of data.
        num_iterations: Number of training iterations.
        device: Device to run the training on ('cpu' or 'cuda').
        save_path: Path to save the checkpoint after training.
    """
    run_save_path = join(save_path, run_name)
    os.makedirs(run_save_path, exist_ok=True)
    step = 0
    sum_loss = 0.0
    log_steps = 50
    total_steps = min(max_steps, dataloader.num_batches)

    with Progress() as progress:
        task = progress.add_task("[green]Training...", total=total_steps)
        
        model.train()
        
        for inputs, targets in dataloader.iter_batch():
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = cross_entropy_loss(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            sum_loss += loss.item()
            
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            progress.update(task, advance=1)

            if step % log_steps == 0:
                lr = scheduler.get_last_lr()
                avg_loss = sum_loss / log_steps
                progress.log(f"[blue]Step {step}/{total_steps}, Avg_Loss: {avg_loss : .6f}, LR: {lr : .6f}[/blue]")
                wandb.log({     
                    "train_loss": loss.item(),
                    "learning_rate": lr
                })
                sum_loss = 0.0

            if step % save_steps == 0:
                progress.log(f"[bold green]Saving checkpoint at step {step}...[/bold green]")
                save_checkpoint(model, optimizer, 0, join(run_save_path, f"ckpt_{step}.pth"))
            
            if step >= max_steps:
                progress.log(f"[bold green]Reached max steps {max_steps}. Stopping training.[/bold green]")
                save_checkpoint(model, optimizer, 0, join(run_save_path, f"ckpt_{step}.pth"))
                break

if __name__ == "__main__":
    from argparse import ArgumentParser

    np.random.seed(42)

    parser = ArgumentParser(description="Train a Transformer Language Model")
    parser.add_argument("--config_path","-c", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # 训练设置
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # 准备数据集/Tokenize
    if not os.path.exists(config["dataset"]["tokenized_train_file"]):
        print("[red]Tokenized train file not found. Tokenizing dataset...")
        print("[bold blue]Loading tokenizer...")
        tokenizer = BPETokenizer(
            vocab = config["tokenizer"]["vocab_file"],
            merges= config["tokenizer"]["merges_file"],
            special_tokens=config["tokenizer"]["special_tokens"]
        )
        
        print("[bold blue]Tokenizing dataset...")

        tokenized_data = tokenizer.parallel_tokenize_txt(config["dataset"]["train_file"],4)
        
        # origin_dataset = open(config["dataset"]["train_file"], "r", encoding="utf-8")
        # tokenized_data = []
        # for token in track(tokenizer.encode_iterable(origin_dataset), description="Tokenizing dataset..."):
        #     tokenized_data.append(token)
        tokenized_data = np.array(tokenized_data, dtype=np.short)
        
        print(f"[green]Tokenization completed with {len(tokenized_data)} tokens. Saving tokenized data...")
        with open(config["dataset"]["tokenized_train_file"], "wb") as f:
            np.save(f, tokenized_data, allow_pickle=True)
    

    # dataset = np.memmap(config["dataset"]["tokenized_train_file"], dtype=np.short, mode='r')
    # 这次先全部加载到内存中
    print("[bold]Loading tokenized dataset from [/bold]" + config["dataset"]["tokenized_train_file"])
    dataset = np.load(config["dataset"]["tokenized_train_file"], allow_pickle=True)
    print(f"[green]Tokenized dataset has {len(dataset)} tokens.")

    # 设置训练参数
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please set device to 'cpu' or check your CUDA installation.")

    os.makedirs(config["training"]["save_path"], exist_ok=True)
    dataloader = DataLoader(
        dataset = dataset, 
        batch_size=config["training"]["batch_size"],
        context_length=config["model"]["context_length"],
        device=device
    )

    if config["training"]["dtype"] == "bfloat16":
        assert device == "cuda", "bfloat16 dtype requires CUDA device"
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = TransformerLM(device=device, dtype=dtype, **config["model"])
    model.to(device)

    if config["training"]["optimizer"] == "adamW":
        optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    else:
        raise NotImplementedError(f"Optimizer {config['training']['optimizer']} is not implemented.")

    if config["training"]["lr_scheduler"]["type"] == "cosine_annealing":
        total_steps = min(config["training"]["max_steps"], dataloader.num_batches)
        max_lr = config["training"]["learning_rate"]
        min_lr = config["training"]["lr_scheduler"]["min_learning_rate"]
        warmup_steps = total_steps * config["training"]["lr_scheduler"]["warmup_ratio"]
        scheduler = lr_scheduler(
            optimizer=optimizer,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=total_steps
        )
        scheduler.step()  # Initialize the scheduler
    else:
        raise NotImplementedError(f"Learning rate scheduler {config['training']['lr_scheduler']} is not implemented.")

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[bold blue]Starting training run:[/bold blue] {run_name}")

    wandb.init(
        project="cs336_assignment1",
        name=run_name,
        mode="offline",
        config=config
    )
    wandb.watch(model, log="all", log_freq=100)

    # 训练循环
    train_loop(
        run_name = run_name,
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler,
        dataloader = dataloader, 
        max_steps=config["training"]["max_steps"],
        save_steps=config["training"]["save_steps"],
        save_path=config["training"]["save_path"]
    )