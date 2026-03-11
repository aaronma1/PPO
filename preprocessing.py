import torch
import numpy as np


class BasePreprocessor:

    def __init__(self, device="cpu"):
        self.device = device


    def __call__(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()
        return obs

    def process_batch(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch).float()
        return batch
    

class InputNormalizingPreprocessor:

    def __init__(self, shape=105, device="cpu", clip=10.0):
        self.device = device
        self.clip = clip
        
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4  # Start small to avoid div by zero
        self.max_count = 1_000_000.0

    def __call__(self, obs, update=True):
        """
        Called in the acting loop takes an observation, updates the running mean and variance, and returns a normalized version of the observation.
        obs: anything
        update: whether to update the running mean and variance with this observation 
        """

        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float().to(self.device)
        else:
            obs.to(self.device)
        obs = torch.atleast_2d(obs)

        if update:
            batch_mean = torch.mean(obs, dim=0)
            batch_var = torch.var(obs, dim=0, unbiased=False)
            batch_count = obs.shape[0]

            # mean tracking
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            self.mean += delta * (batch_count / total_count)

            # variance tracking
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + torch.square(delta) * (self.count * batch_count / total_count)
            self.var = M2 / total_count

            self.count = min(total_count, self.max_count)

        std = torch.sqrt(self.var + 1e-8)
        normalized = (obs - self.mean) / std
        return torch.clamp(normalized, -self.clip, self.clip)
   
    def process_batch(self, batch):
        """
        Called in gradient procesing loop to normalize the observations. 
        batch: tensor of shape (batch_size, obs_dim)
        returns: normalized batch of shape (batch_size, obs_dim) on the same device as the input batch
        """
        m1 = self.mean.to(batch.device)
        v1 = self.var.to(batch.device)
        std = torch.sqrt(v1 + 1e-8)
        normalized = (batch - m1) / std
        return torch.clamp(normalized, -self.clip, self.clip)

class BaseRewardPhi:
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, reward, terminated, truncated):
        return reward

class RollingRewardScaler:
    # scale rewards in episode so that they 
    def __init__(self, gamma=0.99, epsilon=1e-8, device="cpu"):
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        
        self.ret = torch.zeros(1, device=device)
        self.var_tracker = torch.ones(1, device=device)
        self.mean_tracker = torch.zeros(1, device=device)
        self.count = 1e-4

    def __call__(self, reward, terminated, truncated):
        """
        Docstring for __call__
        :param reward: Vector of shape (batch_size,) or (batch_size, 1) or scalar if vector must be on cpu
        :param terminated: Vector of same shape as reward, 
        :param truncated: Vector of same shape as reward, 
        returns: scaled reward of same shape as input reward, on the same device as the input reward

        must be called sequentially on rewards in episode order for the same episode, and must be called on all rewards in an episode (i.e. can't skip rewards) for the return tracking to be correct.
        """
        self.ret = self.ret * self.gamma + reward
        batch_count = 1
        delta = self.ret - self.mean_tracker
        new_count = self.count + batch_count
        self.mean_tracker += delta * (batch_count / new_count)
        m_a = self.var_tracker * self.count
        M2 = m_a + delta * (self.ret - self.mean_tracker)
        self.var_tracker = M2 / new_count
        self.count = new_count
        std = torch.sqrt(self.var_tracker + self.epsilon)
        scaled_reward = reward / std
        
        if terminated or truncated:
            self.ret = 0.0
            
        return scaled_reward