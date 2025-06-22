# train/trainer.py

import torch
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, policy, optimizer, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, batch_size=64):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.batch_size = batch_size

    def update(self, buffer):
        for _ in range(self.epochs):
            for batch in buffer.get_batches(self.batch_size):
                obs = batch['obs']
                actions = batch['actions']
                old_logprobs = batch['logprobs']
                advantages = batch['advantages']
                returns = batch['returns']

                values, logprobs, entropy = self.policy.evaluate(obs, actions)

                # 计算比值 r_theta
                ratio = torch.exp(logprobs - old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 保证维度一致，避免广播 warning
                values = values.squeeze()  # (B, 1) -> (B,) 如果必要

                value_loss = F.mse_loss(values, returns)
                entropy_loss = entropy.mean()

                # 总损失
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item()
        }
