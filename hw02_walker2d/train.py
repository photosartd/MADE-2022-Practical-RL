import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 5e-4
CRITIC_LR = 2e-4

CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 1024

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 3000

device = "cuda" if torch.cuda.is_available() else "cpu"

    
def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)
    
    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]
    


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        hid_dim = state_dim // 2
        self.model = nn.Sequential(
            nn.Linear(state_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, action_dim)
        )
        self.sigma = nn.Parameter(torch.ones(action_dim))
        
    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        distrib = self.get_action_distribution(state)
        return torch.exp(distrib.log_prob(action).sum(-1))
        
    def act(self, state):
        distrib = self.get_action_distribution(state)
        pure_action = distrib.sample()
        action = torch.tanh(pure_action)
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        return action, pure_action, distrib

    def get_action_distribution(self, state):
        mu = self.model(state)
        #batch_size x action_dim
        distrib = Normal(mu, self.sigma)
        return distrib
        
        
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        
    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)

    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj] # Turn a list of trajectories into list of transitions (s, a, p, v, adv)
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advnatage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        
        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE) # Choose random batch
            s = torch.tensor(state[idx]).float().to(device)
            a = torch.tensor(action[idx]).float().to(device)
            op = torch.tensor(old_prob[idx]).float().to(device) # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float().to(device).unsqueeze(-1) # Estimated by lambda-returns 
            adv = torch.tensor(advantage[idx]).float().to(device) # Estimated by generalized advantage estimation 
            
            new_p = self.actor.compute_proba(s, a)
            new_v = self.critic.get_value(s)

            critic_loss = F.mse_loss(new_v, v)
            #print(critic_loss.shape)
            #print(critic_loss)
            ratios = new_p / op
            h1 = ratios * adv
            h2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * adv
            actor_loss = - torch.min(h1, h2).mean()
            #print(actor_loss.shape)
            #print(actor_loss)
            #entropy_loss = - self.actor.get_action_distribution(s).entropy().mean()
            #print(entropy_loss.shape)
            #print(entropy_loss)

            loss = actor_loss + critic_loss
            #print(loss.shape)
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
            self.actor_optim.step()
            self.critic_optim.step()
            
            
    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(device)
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(device)
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        torch.save(self.actor.state_dict(), "agent.pt")


def evaluate_policy(env, agent, episodes=10):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns
   

def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)

if __name__ == "__main__":
    env = make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0
    best_mean_reward = 0
    
    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0
        
        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)        
        
        if (i + 1) % (ITERATIONS//300) == 0:
            rewards = evaluate_policy(env, ppo, 5)
            curr_mean_reward = np.mean(rewards)
            print(f"Step: {i+1}, Reward mean: {curr_mean_reward}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
            if curr_mean_reward > best_mean_reward:
                best_mean_reward = curr_mean_reward
                ppo.save()
