import torch
from torch import nn

from time import time

class ToyModel(nn.Module):  
    def __init__(self, in_features: int, out_features: int): 
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False) 
        self.ln = nn.LayerNorm(10) 
        self.fc2 = nn.Linear(10, out_features, bias=False) 
        self.relu = nn.ReLU()
    def forward(self, x): 
        x = self.relu(self.fc1(x))
        print("After fc1:", x.dtype)
        x = self.ln(x)
        print("After ln:", x.dtype)
        x = self.fc2(x)
        return x

def model_run_demo():
    t1 = time()
    x = torch.randn(2, 20, dtype=torch.float32) .to("cuda")
    model = ToyModel(20, 10).to("cuda")
    target = torch.randn(2, 10, dtype=torch.float32).to("cuda")
    print("Model weight dtypes:", model.fc1.weight.dtype)
    y = model(x)
    print("Output dtype:", y.dtype)
    loss = nn.MSELoss()(y, target)
    print("Loss dtype:", loss.dtype)
    loss.backward()
    torch.cuda.synchronize()
    print("Gradient dtype:", model.fc1.weight.grad.dtype)
    print("Time taken:", time() - t1)

if __name__ == "__main__":
    print("Running without autocast:")
    model_run_demo()

    with torch.autocast(dtype=torch.bfloat16, device_type='cuda'): 
        print("Running with autocast to bfloat16:")
        model_run_demo()