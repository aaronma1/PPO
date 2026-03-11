import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def save_and_plot_experiment(episode_returns, experiment_name="PPO_Research"):
    # 1. Create directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"./results/{experiment_name}_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    
    # 2. Save raw data (Numpy and JSON for portability)
    np.save(f"{folder}/returns.npy", np.array(episode_returns))
    with open(f"{folder}/metadata.json", "w") as f:
        json.dump({"experiment": experiment_name, "total_episodes": len(episode_returns)}, f)
    
    # 3. Generate Plot
    plt.figure(figsize=(10, 6))
    
    # Raw returns (lighter color)
    plt.plot(episode_returns, alpha=0.3, color='#1f77b4', label='Raw Return')
    
    # Rolling average (100 episodes) to see the trend
    if len(episode_returns) > 100:
        rolling_mean = np.convolve(episode_returns, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(episode_returns)), rolling_mean, color='#d62728', linewidth=2, label='100-Ep Moving Avg')
    
    plt.title(f"Training Performance: {experiment_name}", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Return", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Save high-res figure for your ICML draft
    plt.savefig(f"{folder}/training_curve.png", dpi=300)
    plt.show()
    print(f"Data and plots saved to: {folder}")

# Usage:
# episode_return_list = [your_actual_returns]
# save_and_plot_experiment(episode_return_list, "Ant_0.999_NEO_v1")