import torch
import numpy as np
from typing import Iterable, Tuple

class DataLoader():
    def __init__(self, dataset: np.array, batch_size: int, context_length: int, device: str):
        self.dataset = dataset
        self.context_length = context_length
        self.device = device

        self.data_length = dataset.shape[0]
        self.batch_size = batch_size
        self.num_batches = (self.data_length - self.context_length) // batch_size

    def generate_random_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a single random batch of input sequences and their corresponding next-token targets.
        
        Returns:
            Tuple of (inputs, targets) tensors, both on the specified device and of shape (batch_size, context_length).
        """
        
        # Initialize tensors on the specified device
        inputs = torch.zeros((self.batch_size, self.context_length), dtype=torch.long, device=self.device)
        targets = torch.zeros((self.batch_size, self.context_length), dtype=torch.long, device=self.device)
        
        # Fill tensors with sequences
        indices = np.random.choice(np.arange(self.data_length - self.context_length), size=self.batch_size, replace=False)
        for i, start_idx in enumerate(indices):
            end_idx = start_idx + self.context_length
            inputs[i] = torch.tensor(self.dataset[start_idx : end_idx])
            targets[i] = torch.tensor(self.dataset[start_idx + 1 : end_idx + 1])
        
        return inputs, targets

    def iter_batch(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns an iterable generator that yields batches of inputs and targets.
        """
        indices = np.arange(self.data_length - self.context_length)
        np.random.shuffle(indices)
        
        pos = 0
        for _ in range(self.num_batches):
            inputs = torch.zeros((self.batch_size, self.context_length), dtype=torch.long, device=self.device)
            targets = torch.zeros((self.batch_size, self.context_length), dtype=torch.long, device=self.device)

            for i in range(self.batch_size):
                start_idx = indices[pos]
                end_idx = start_idx + self.context_length
                inputs[i] = torch.tensor(self.dataset[start_idx : end_idx])
                targets[i] = torch.tensor(self.dataset[start_idx + 1 : end_idx + 1])
                pos += 1
            
            yield inputs, targets