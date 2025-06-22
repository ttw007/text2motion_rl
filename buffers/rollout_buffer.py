# buffers/rollout_buffer.py

import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, action_dim, device='cpu'):
        self.obs_buf = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards_buf = np.zeros(buffer_size, dtype=np.float32)
        self.dones_buf = np.zeros(buffer_size, dtype=np.float32)
        self.values_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logprobs_buf = np.zeros(buffer_size, dtype=np.float32)
        self.advantages_buf = np.zeros(buffer_size, dtype=np.float32)
        self.returns_buf = np.zeros(buffer_size, dtype=np.float32)

        self.device = device
        self.ptr = 0
        self.max_size = buffer_size

    def store(self, obs, action, reward, done, value, log_prob):
        if self.ptr >= self.max_size:
            return  # 超过容量则忽略
        self.obs_buf[self.ptr] = obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.dones_buf[self.ptr] = done
        self.values_buf[self.ptr] = value
        self.logprobs_buf[self.ptr] = log_prob
        self.ptr += 1

    def compute_gae(self, gamma=0.99, lam=0.95):
        adv = 0
        for t in reversed(range(self.ptr)):
            next_value = self.values_buf[t + 1] if t + 1 < self.ptr else 0
            delta = self.rewards_buf[t] + gamma * next_value * (1 - self.dones_buf[t]) - self.values_buf[t]
            adv = delta + gamma * lam * (1 - self.dones_buf[t]) * adv
            self.advantages_buf[t] = adv
            self.returns_buf[t] = adv + self.values_buf[t]  # 添加回报计算

        # 标准化 advantages
        self.advantages_buf = (self.advantages_buf - self.advantages_buf.mean()) / (self.advantages_buf.std() + 1e-8)

    def get_batches(self, batch_size):
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                "obs": torch.tensor(self.obs_buf[batch_idx], dtype=torch.float32).to(self.device),
                "actions": torch.tensor(self.actions_buf[batch_idx], dtype=torch.float32).to(self.device),
                "returns": torch.tensor(self.returns_buf[batch_idx], dtype=torch.float32).to(self.device),
                "advantages": torch.tensor(self.advantages_buf[batch_idx], dtype=torch.float32).to(self.device),
                "logprobs": torch.tensor(self.logprobs_buf[batch_idx], dtype=torch.float32).to(self.device),
            }

    def clear(self):
        self.ptr = 0
