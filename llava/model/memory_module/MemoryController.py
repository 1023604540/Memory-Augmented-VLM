import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class EpisodicMemoryController:
    def __init__(self, mem_slots=32, mem_patch=196, mem_dim=896, device='cpu', dtype=torch.float16):
        # Initialize memory matrices (keys and values could be same in this simple case)
        self.device = device
        self.original_dtype = dtype
        self.compute_dtype = torch.float32
        self.mem_keys = torch.zeros((mem_slots, mem_patch, mem_dim), device=device, dtype=self.compute_dtype)
        self.mem_vals = torch.zeros((mem_slots, mem_patch, mem_dim), device=device, dtype=self.compute_dtype)
        self.capacity = mem_slots
        self.mem_patch = mem_patch
        self.mem_dim = mem_dim
        # Track number of written slots or a pointer for FIFO
        self.next_idx = 0
        # Optionally, an initial covariance or other needed for least-squares updates
        # self.cov_inv = np.eye(mem_dim) * alpha  (if using advanced update rules)

    def retrieve_memory(self, query_vec):
        # Zr = (Zq*M† + noise)M
        original_dtype = query_vec.dtype
        query_vec = query_vec.to(self.compute_dtype)  # (Nq, D)
        cur_memory = self.mem_keys[:self.next_idx].flatten(0, 1)  # (N*P, D)
        try:
            # Try computing the pseudoinverse with regularization
            reg = 1e-4
            eye = torch.eye(cur_memory.shape[1], device=cur_memory.device, dtype=cur_memory.dtype)
            memory_inv = torch.linalg.pinv(cur_memory + reg * eye)
        except Exception as e:
            print(f"[Warning] Failed to compute pseudoinverse: {e}, Memory to be inversed: {cur_memory.shape} ")
            print("[Info] Falling back to zero memory response.")
            # Fallback: use zeroed memory or skip retrieval
            zero_response = torch.zeros(query_vec.shape[0], query_vec.shape[1], device=query_vec.device)
            return zero_response.to(original_dtype)

        temp = query_vec @ memory_inv  # (Nq, N*P)
        temp_add_noise = self.add_noise(temp, sigma=0.001)  # (Nq, N*P)
        Z = temp_add_noise @ cur_memory  # (Nq, D)
        return Z.to(original_dtype)

    def integrate(self, old_memory, new_memory):
        # M^ =(ZξM0†)†Zξ
        old_memory = old_memory.to(self.compute_dtype)
        new_memory = new_memory.to(self.compute_dtype)

        if new_memory.dim() == 3:
            new_memory = new_memory.flatten(0, 1)
        if old_memory.dim() == 3:
            old_memory = old_memory.flatten(0, 1)

        Z = self.add_noise(new_memory, sigma=0.001)  # (N*P, D)
        # print(f"1, {time.time()-write_time}, z shape: {Z.shape}")
        M0_inverse = torch.linalg.pinv(old_memory)  # (D, N*P)
        # print(f"2, {time.time()-write_time}, M0_inverse shape: {M0_inverse.shape}")
        Temp = new_memory @ M0_inverse  # (N*P, N*P)
        # print(f"3, {time.time()-write_time}")
        # Temp_inverse = torch.linalg.pinv(Temp)  # (N*P, N*P)
        M_hat = torch.linalg.lstsq(Temp, Z).solution
        # print(f"4, {time.time()-write_time}, M_hat shape: {M_hat.shape}")
        #M_hat = Temp_inverse @ Z  # (N*P, D)
        # print(f"5, {time.time()-write_time}")
        self.mem_keys = M_hat.view(self.capacity, self.mem_patch, self.mem_dim)
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
        # write_time = time.time()
        episode_vec = episode_vec.to(self.compute_dtype)

        if self.next_idx >= self.capacity:
            # Memory full: integrate into existing slots (distributed update).
            # For simplicity, find the closest existing key and replace it (or merge).
            self.integrate(self.mem_keys, episode_vec)
            return

        input_len = len(episode_vec)
        available_space = self.capacity - self.next_idx

        if input_len <= available_space:
            for idx in range(input_len):
                self.mem_keys[self.next_idx + idx] = episode_vec[idx]
                self.mem_vals[self.next_idx + idx] = episode_vec[idx]
            self.next_idx += input_len
        else:
            for idx in range(available_space):
                self.mem_keys[self.next_idx + idx] = episode_vec[idx]
                self.mem_vals[self.next_idx + idx] = episode_vec[idx]
            # print(f"mid, {time.time() - write_time}")
            self.next_idx = self.capacity
            self.integrate(self.mem_keys, episode_vec[available_space:])
        # print(f"end, {time.time() - write_time}")
        return

        # End of write_memory


