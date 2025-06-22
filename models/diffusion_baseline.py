import torch
import torch.nn as nn

class DiffusionBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=72):
        super(DiffusionBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_feat):
        return self.net(text_feat)