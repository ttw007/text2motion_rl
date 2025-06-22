import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        hidden_size = 256

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.actor_head = nn.Linear(hidden_size, act_dim)
        self.critic_head = nn.Linear(hidden_size, 1)

        self.log_std = nn.Parameter(torch.zeros(act_dim))  # shape: (263,)

    def actor(self, obs):
        x = self.net(obs)
        mean = self.actor_head(x)
        mean = torch.tanh(mean) * 1.0
        std = torch.exp(self.log_std).expand_as(mean).clamp(min=1e-3, max=1.0)  # 保证 std > 0
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def critic(self, obs):
        x = self.net(obs)
        value = self.critic_head(x)
        return value

    def evaluate(self, obs, act):
        x = self.net(obs)
        mean = self.actor_head(x)
        dist = torch.distributions.Normal(mean, torch.ones_like(mean) * 0.1)
        log_prob = dist.log_prob(act).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.critic_head(x)
        return value, log_prob, entropy