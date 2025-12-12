import torch
import torch.nn as nn
import torch.nn.functional as F

class A3CNet(nn.Module):
    
    def __init__(self, num_actions: int):
        super().__init__()

        obs_dim = 4  

        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        self.policy = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.policy(x)
        value = self.value(x)
        return logits, value