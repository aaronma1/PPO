import gymnasium as gym
from pathlib import Path
import torch
import numpy as np


import agent_environment
from buffer import FastBuffer
from ppo import PPOAgent
from pppo import PPPOAgent
from networks import PolicyNet, CriticNet
from preprocessing import InputNormalizingPreprocessor, RollingRewardScaler


# made by gemini
def save_episode_data(episode_returns, episode_end_steps, experiment_name, run_idx):
    """
    Saves returns to data/experiment_name/run_{i}.txt
    Creates folders automatically if they don't exist.
    """
    # 1. Define the directory and file path
    # Path handles / vs \ automatically (great for Mac/Linux portability)

    dir_path = Path(f"data/{experiment_name}")
    file_path = dir_path / f"run_{run_idx}.txt"

    # 2. Create the folders (parents=True allows nested creation)
    dir_path.mkdir(parents=True, exist_ok=True)

    # 3. Write returns to file (one per line)
    with open(file_path, "w") as f:
        for (r, t) in zip(episode_returns, episode_end_steps):
            f.write(f"{r}, {t}\n")
            
    print(f"Successfully saved {len(episode_returns)} episodes to: {file_path}")


if __name__ == '__main__':
    # agents = [PPPPOAgent, PPOAgent]
    agents = [PPPOAgent]
    TRAIN_STEPS=20_000_000
    envs = [("Humanoid-v5", 17, 348)]
    # envs = [("Humanoid-v5", 17, 348), ("Ant-v5", 8, 105), ("Walker2d-v5", 6, 17)]
    agent_to_str = {PPPOAgent: "PPPO", PPOAgent: "PPO"}

    episode_reward_list = []

    for env_name, action_dim, observation_dim in envs:
        for seed in range(1):
            print(f"Running PPPO on {env_name} with seed {seed}")
            env = gym.make(env_name, render_mode="rgb_array")
            actor = PolicyNet(
                obs_dim = observation_dim,
                act_dim = action_dim,
                action_scale = 1.0,
                hidden_sizes = [
                    256,
                    256,
                ],
                activation=torch.nn.Tanh,
                action_init_std=np.log(0.35)
            ).to("cuda")

            critic = CriticNet(
                input_dim=observation_dim,
                hidden_sizes = [
                    512,
                    512,
                ],
                activation=torch.nn.Tanh
            ).to('cuda')

            buffer = FastBuffer(observation_dim, action_dim, 4097)

            agent = PPPOAgent(
                actor,
                critic,
                buffer,
                gamma = 0.99,
                gae_lambda=0.95,
                actor_alpha = 3e-4,
                critic_alpha=3e-4,
                ppo_clip=0.2,
                entropy_bonus=0.001,
                ppo_early_stopping_kl=0.04,

                actor_final_alpha=1e-4,
                critic_final_alpha=1e-4,
                falloff_steps=450,

                use_updated_critic_adv=True,
                minibatch_adv_normalization=False,
                critic_loss_clip=0.2,
                actor_update_iters=10,
                critic_update_iters=20,
                actor_batch_size=2048,
                critic_batch_size=2048,

                transitions_per_gradient_update=4096,
                input_preprocessor=InputNormalizingPreprocessor(shape=observation_dim),
                reward_phi = RollingRewardScaler(),
            )


            episode_rewards, episode_end_steps, agent_all_info = agent_environment.agent_environment_step_loop(env, agent, TRAIN_STEPS, debug=True, record=True)
            episode_reward_list.append(episode_rewards)

            save_episode_data(episode_rewards, episode_end_steps, "{env_name}_ppo_default", seed)
            

    