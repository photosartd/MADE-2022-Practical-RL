import random
import numpy as np
import os
import torch

from .train import Actor


class Agent:
    def __init__(self):
        self.model = Actor(22, 6)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt"))
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            return self.model.act(state)[0]

    def reset(self):
        pass

