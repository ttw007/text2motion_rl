import torch
import torch.nn as nn
from models.actor_critic import ActorCritic

class PolicyWrapper(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor_critic = ActorCritic(obs_dim, act_dim)

    def act(self, obs):
        action, log_prob = self.actor_critic.actor(obs)
        value = self.actor_critic.critic(obs)
        return action, value, log_prob

    def evaluate(self, obs, action):
        return self.actor_critic.evaluate(obs, action)
