import torch._dynamo
torch._dynamo.config.optimize_ddp = False
from llava.train.train import train

if __name__ == "__main__":
    train()
