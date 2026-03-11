import gymnasium
import numpy as np
import torch 
import numpy as np
import tqdm


class FastBuffer:

    def __init__(self, state_shape, action_shape, capacity, store_logp = False, device='cuda'):
        print(state_shape, action_shape)
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.len = 0
        self.capacity = capacity
        self.device = "cuda"

        self.states = torch.zeros((self.capacity, self.state_shape),  dtype=torch.float32 ).pin_memory()
        self.actions = torch.zeros((self.capacity, self.action_shape), dtype=torch.float32).pin_memory()
        self.next_states = torch.zeros((self.capacity, self.state_shape), dtype=torch.float32 ).pin_memory()
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).pin_memory()
        self.terminated = torch.zeros((self.capacity, 1), dtype=torch.float32).pin_memory()
        self.truncated = torch.zeros((self.capacity, 1), dtype=torch.float32).pin_memory()
        # gpu pinned 
        self.gpu_states = torch.zeros((self.capacity, self.state_shape), device=self.device, dtype=torch.float32 )
        self.gpu_actions = torch.zeros((self.capacity, self.action_shape),device=self.device, dtype=torch.float32)
        self.gpu_next_states = torch.zeros((self.capacity, self.state_shape), device=self.device, dtype=torch.float32 )
        self.gpu_rewards = torch.zeros((self.capacity, 1), device=self.device, dtype=torch.float32)
        self.gpu_terminated = torch.zeros((self.capacity, 1), device=self.device, dtype=torch.float32)
        self.gpu_truncated = torch.zeros((self.capacity, 1), device=self.device, dtype=torch.float32)
        self.reset()


    def __len__(self):
        return self.len

    def reset(self):
        self.len = 0
    

    def push(self, state, action, reward, next_state, terminated, truncated, logp=None):
        self.states[self.len, :] = torch.from_numpy(state).float()
        self.actions[self.len, :] = torch.from_numpy(action).float()
        self.next_states[self.len, :] = torch.from_numpy(next_state).float()
        self.rewards[self.len][0] = reward
        self.terminated[self.len][0] = terminated
        self.truncated[self.len][0] = truncated
        self.len += 1
    
    def flush(self):
        self.gpu_states[:self.len].copy_(self.states[:self.len], non_blocking=True)
        self.gpu_actions[:self.len].copy_(self.actions[:self.len], non_blocking=True)
        self.gpu_next_states[:self.len].copy_(self.next_states[:self.len], non_blocking=True)
        self.gpu_rewards[:self.len].copy_(self.rewards[:self.len], non_blocking=True)
        self.gpu_terminated[:self.len].copy_(self.terminated[:self.len], non_blocking=True)
        self.gpu_truncated[:self.len].copy_(self.truncated[:self.len], non_blocking=True)

    
        rtn =  self.gpu_states[:self.len], self.gpu_actions[:self.len], self.gpu_rewards[:self.len], self.gpu_next_states[:self.len], self.gpu_terminated[:self.len], self.gpu_truncated[:self.len]
        self.len = 0
        return rtn



class ReplayBuffer:
    def __init__(self, device="cuda"):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminated = []
        self.truncated = []

        self.device = device

    def __len__(self):
        return len(self.terminated)
    

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminated = []
        self.truncated = []

        
    def push(self, state, action, reward, next_state, terminated, truncated):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    # #TOOD: implement subsample ratio and shuffle if needed
    def flush(self, shuffle = False, subsample_ratio=1.0):

        size = len(self)

        states = torch.from_numpy(np.array(self.states).reshape((size, -1))).to(self.device).float()
        actions = torch.from_numpy(np.array(self.actions).reshape((size, -1))).to(self.device).float()
        next_states = torch.from_numpy(np.array(self.next_states).reshape((size, -1))).to(self.device).float()

        rewards = torch.tensor(self.rewards, device=self.device, dtype=torch.float32).reshape(-1, 1)
        terminated = torch.tensor(self.terminated, device=self.device, dtype=torch.float32).reshape(-1,1)
        truncated = torch.tensor(self.truncated, device=self.device, dtype=torch.float32).reshape(-1,1)
        self.reset()
        return states, actions, rewards, next_states, terminated, truncated

