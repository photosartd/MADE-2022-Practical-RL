from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QModel(nn.Module):
    def __init__(self, observation_space, action_space, hidden_space=64, seed=SEED):
        super(QModel, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_space = hidden_space
        
        self.linear1 = nn.Linear(observation_space, hidden_space)
        self.linear2 = nn.Linear(hidden_space, hidden_space)
        self.linear3 = nn.Linear(hidden_space, action_space)
        
    def forward(self, state):
        res = F.relu(self.linear1(state))
        res = F.relu(self.linear2(res))
        return self.linear3(res)
    
class ReplayBuffer:
    def __init__(self, maxlen=10000, seed=SEED):
        
        self.buffer = deque(maxlen=maxlen)
        self.Transition = namedtuple("Transition", field_names=["state", "action", "next_state", "reward", "done"])
        self.seed = random.seed(seed)
        
    def add(self, transition):
        transition = self.Transition(*transition)
        self.buffer.append(transition)
        
    def sample(self, batch_size=512):
        transitions = random.sample(self.buffer, k=batch_size)
        
        #size of all: [batch_size, 1]
        states = torch.from_numpy(np.vstack([t.state for t in transitions if t is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in transitions if t is not None])).long().to(device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in transitions if t is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in transitions if t is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([t.done for t in transitions if t is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, next_states, rewards, dones)
    
    def __len__(self):
        return len(self.buffer)
    
class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0 # Do not change
        
        self.model = QModel(state_dim, action_dim).to(device)
        self.target_model = QModel(state_dim, action_dim).to(device)
        self.optimizer = Adam(self.model.parameters())
        self.buffer = ReplayBuffer()

    def consume_transition(self, transition):
        self.buffer.add(transition)
        
    def sample_batch(self):
        return self.buffer.sample()
        
    def train_step(self, batch):
        states, actions, next_states, rewards, dones = batch
        
        q_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + GAMMA * q_next * (1 - dones)
        q_pred = self.model(states).gather(1, actions)
        
        self.optimizer.zero_grad()
        loss = F.mse_loss(q_pred, q_target)
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(local_param.data)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.model.eval()
        with torch.no_grad():
            action = np.argmax(self.model(state).cpu().data.numpy())
        self.model.train()

        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model.state_dict(), "agent.pt")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    best_mean_reward = -1000
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            mean_reward = np.mean(rewards)
            print(f"Step: {i+1}, Reward mean: {mean_reward}, Reward std: {np.std(rewards)}")
            if mean_reward > best_mean_reward:
                dqn.save()
