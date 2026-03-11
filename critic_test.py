





import gymnasium as gym
import torch



import agent_environment
from buffer import ReplayBuffer
from ppo import PPOAgent
from networks import trpo_policy_network, critic_net


class actor_net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([0.0, 1.0], device="cuda"))

    
    def forward(self, x):
        x = x.reshape(-1, 2)
        x = x[:, 1]/(torch.abs(x[:, 1]) + 1e-4)

        return x

    def sample_action(self, obs):
        return self.forward(obs)


    def compute_log_prob(self, obs, action):
        return torch.zeros_like(action, device="cuda"), torch.zeros_like(action, device="cuda"), torch.zeros_like(action, device="cuda")

    





def input_preprocessor(obs):
    if not isinstance(obs, torch.Tensor):
        obs = torch.from_numpy(obs)
    obs = obs.to("cuda")
    obs = obs *  torch.tensor([1.0, 1.0], device="cuda")
    return obs

def reward_phi(r):
    return r * 0.1


if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")


    action_size = 1
    observation_size = 2


    agents = [PPOAgent]
    agent_to_str = {PPOAgent: "PPO"}


    episode_reward_list = []
    for alg in agents:
        for seed in range(1):
            actor = actor_net(
            ).to("cuda")

            print(actor(torch.tensor([[0,0.07]], device="cuda")))

            critic = critic_net(
                input_dim=observation_size,
                hidden_sizes = [
                    128,
                    128
                ],
            ).to('cuda')

            buffer = ReplayBuffer()


            agent = alg(
                actor,
                critic,
                buffer, 
                gamma = 0.99, 
                actor_alpha = 1e-4,
                critic_alpha=1e-4,
                transitions_per_gradient_update = 100,
                input_preprocessor=input_preprocessor,
                reward_phi = reward_phi
            )


    
                 
                 
            episode_rewards = agent_environment.agent_environment_step_loop(env, agent, 10_000, debug=True)
            episode_reward_list.append(episode_rewards)

