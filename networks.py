from collections import OrderedDict

import torch
import numpy as np
        
class MuSigmaHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim, action_init_std=-0.5):
        super().__init__()

        self.mu_head = torch.nn.Linear(input_dim, output_dim)
        
        # 1. State-independent log standard deviation (Free Parameter)
        # We use log_std because it can take any value from -inf to +inf without breaking
        self.log_std = torch.nn.Parameter(torch.ones(output_dim)*action_init_std)

        self.log_std_head = torch.nn.Linear(input_dim, output_dim)

        # 2. Correct initialization for PPO
        def _init_last_layer(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=0.01)
                torch.nn.init.constant_(m.bias, 0)
        self.mu_head.apply(_init_last_layer)
        # 2. Correct initialization for resudiual log_std
        def _init_last_layer(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.weight, 0)
                torch.nn.init.constant_(m.bias, 0)
        self.log_std_head.apply(_init_last_layer)

    
    def forward(self, x):
        mu = self.mu_head(x)
        # Expand the log_std parameter to match the batch size of 'x'
        # Then exponentiate to get standard deviation
        sigma = torch.exp(self.log_std.expand_as(mu) + self.log_std_head(x))
        return mu, sigma


class PolicyNet(torch.nn.Module):

    def __init__(self, obs_dim, act_dim, action_scale = 1.0,
                hidden_sizes = (128, 128),
                output_activation = MuSigmaHead, 
                activation=torch.nn.Tanh,
                tanh_squashing=False,
                action_init_std = 1.0,
                ):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(last_dim, h))
            layers.append(torch.nn.LayerNorm(h))
            layers.append(activation())
            last_dim = h
        self.net = torch.nn.Sequential(*layers)
        self.action_scale = action_scale
        self.output_layer = output_activation(last_dim, act_dim, action_init_std=action_init_std)
        self.tanh_squashing=tanh_squashing

        def _init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0)
        self.net.apply(_init_weights)


    def forward(self, obs):
        h = self.net(obs)
        mu, sigma = self.output_layer(h)
        return mu, sigma
    
    def sample_action(self, obs):
        mu, sigma = self.forward(obs)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        if self.tanh_squashing:
            return torch.tanh(action)*self.action_scale
        else:
            return action

    def compute_log_prob(self,obs, actions):
        mu, sigma = self.forward(obs)
        dist = torch.distributions.Normal(mu, sigma)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
         # Inverse tanh (z-space)

        if not self.tanh_squashing:
            logp_z = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            return logp_z, entropy, mu, sigma
        else:
            eps = 1e-6
            unsquashed_actions =  torch.arctanh(torch.clamp(actions / self.action_scale, -1+eps, 1-eps))
            # Log prob
            logp_z = dist.log_prob(unsquashed_actions).sum(dim=-1, keepdim=True)
            # Jacobian correction
            log_det_jacobian = np.log(self.action_scale) - torch.log(1 - (actions/self.action_scale) ** 2 + eps)
            log_det_jacobian = np.log(self.action_scale) - 2.0 * np.log(2.0) - unsquashed_actions - torch.nn.functional.softplus(-2.0 * unsquashed_actions)

            return logp_z - torch.sum(log_det_jacobian, dim=-1, keepdim=True), entropy, mu, sigma

    def get_params(self):
        """Flat vector of all trainable parameters."""
        params = [p.view(-1) for p in self.parameters()]
        return torch.cat(params)

    def set_params(self, new_params):
        """Apply a flat vector of parameters back to the network layers."""
        current_idx = 0
        for p in self.parameters():
            n_params = p.numel()
            p.data.copy_(new_params[current_idx:current_idx + n_params].view(p.size()))
            current_idx += n_params

    def flatten_gradients(self):
        """Extracts gradients and flattens them into a single vector."""
        grads = [p.grad.view(-1) for p in self.parameters() if p.grad is not None]
        return torch.cat(grads)




class CriticNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_sizes = (128, 128), activation = torch.nn.Tanh):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(last_dim, h))
            layers.append(torch.nn.LayerNorm(h))
            layers.append(activation())
            last_dim = h
        layers.append(torch.nn.Linear(last_dim, 1))
        self.net = torch.nn.Sequential(*layers)

        def _init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.constant_(m.bias, 0)
        self.net.apply(_init_weights)

    def forward(self, obs):
        return self.net(obs)
    