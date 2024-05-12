import random
import numpy as np
import os
import torch
from .train import Actor


class Agent:
    def __init__(self):
        self.model = Actor(28, 8)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt"))
        self.model.eval()
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float)
            return np.clip(self.model(state).cpu().numpy(), -1, +1)

    def reset(self):
        pass

