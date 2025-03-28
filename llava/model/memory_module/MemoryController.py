import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class EpisodicMemoryController:
    def __init__(self, mem_slots=32, mem_patch=196, mem_dim=1024, device='cpu', dtype=torch.float16):
        # Initialize memory matrices (keys and values could be same in this simple case)
        self.device = device
        self.mem_keys = torch.zeros((mem_slots, mem_patch, mem_dim), device=self.device, dtype=dtype)
        self.mem_vals = torch.zeros((mem_slots, mem_patch, mem_dim), device=self.device, dtype=dtype)
        self.capacity = mem_slots
        self.mem_dim = mem_dim
        # Track number of written slots or a pointer for FIFO
        self.next_idx = 0
        # Optionally, an initial covariance or other needed for least-squares updates
        # self.cov_inv = np.eye(mem_dim) * alpha  (if using advanced update rules)

    def retrieve_memory(self, query_vec):
        """
        Retrieve memory relevant to the given query vector.
        Returns either a single context vector or a list of top-k memory vectors.
        """
        #  Zr =(Z_q@M†+ξ)M
        query_vec = query_vec.to(self.mem_keys.dtype)
        print(f"query vector: {query_vec.shape}")
        memory_inv = torch.linalg.pinv(self.mem_keys)
        print(f"memory inverse: {memory_inv.shape}")
        temp = query_vec @ memory_inv
        print(f"temp: {temp.shape}")
        temp_add_noise = self.add_noise(temp, sigma=0.001)
        print(f"temp add noise: {temp_add_noise.shape}")
        Z = temp_add_noise @ self.mem_keys
        #print(f"retrieved memory: {Z.shape}")
        return Z

    def integrate(self, old_memory, new_memory):
        # M^ =(ZξM0†)†Zξ
        if new_memory.dim() == 3:
            new_memory = new_memory.flatten(0, 1)
        if old_memory.dim() == 3:
            old_memory = old_memory.flatten(0, 1)
        Z = self.add_noise(new_memory, sigma=0.001)
        print(f"old_memory: {old_memory.shape}")
        M0_inverse = torch.linalg.pinv(old_memory)
        print(f"old_memory inverse: {M0_inverse.shape}")
        Temp = new_memory @ M0_inverse
        print(f"Temp: {Temp.shape}")
        Temp_inverse = torch.linalg.pinv(Temp)
        print(f"Temp inverse: {Temp_inverse.shape}")
        M_hat = Temp_inverse @ Z
        print(f"integrated memory: {M_hat.shape}")
        self.mem_keys = M_hat
        return

    def add_noise(self, x, sigma=0.001):
        """
        Add isotropic Gaussian noise with std=sigma to each element in X.
        """
        if sigma > 0.0:
            return x + sigma * torch.randn_like(x)
        else:
            return x

    def write_memory(self, episode_vec):
        """
        Write a new episode vector into memory (with distributed update if needed).
        """
        if self.next_idx > self.capacity:
            # Memory full: integrate into existing slots (distributed update).
            # For simplicity, find the closest existing key and replace it (or merge).
            self.integrate(self.mem_keys, episode_vec)

        input_len = len(episode_vec)
        if input_len <= self.capacity - self.next_idx:
            for idx in range(input_len):
                self.mem_keys[self.next_idx + idx] = episode_vec[idx]
                self.mem_vals[self.next_idx + idx] = episode_vec[idx]
            self.next_idx += input_len
        else:
            cur_len = self.capacity - self.next_idx
            for idx in range(cur_len):
                self.mem_keys[self.next_idx + idx] = episode_vec[idx]
                self.mem_vals[self.next_idx + idx] = episode_vec[idx]
            self.next_idx = self.capacity
            self.integrate(self.mem_keys, episode_vec[cur_len:])
        return

        # End of write_memory
