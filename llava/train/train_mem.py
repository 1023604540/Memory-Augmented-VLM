from llava.train.train import train

import torch, os
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

if __name__ == "__main__":
    train()
