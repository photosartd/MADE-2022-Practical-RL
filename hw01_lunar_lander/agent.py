import random
import numpy as np
import os
import torch

from .train import QModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self):
        self.model = QModel(8, 4).to(device)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pt"))
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.model.eval()
        with torch.no_grad():
            action = np.argmax(self.model(state).cpu().data.numpy())
        return action

