# import torch._dynamo
# torch._dynamo.config.optimize_ddp = False
import os

print(f"[RANK={os.environ.get('RANK')}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[RANK={os.environ.get('RANK')}] torch.cuda.device_count()={torch.cuda.device_count()}")
from llava.train.train import train

if __name__ == "__main__":
    train()
