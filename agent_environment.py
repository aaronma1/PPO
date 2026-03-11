import tqdm 
import numpy as np
from gymnasium.wrappers import RecordVideo


def agent_environment_step_loop(env, agent, num_steps, debug, record=False ):

    if record:
        def record_trigger(episode_id):
            return episode_id % 300 == 0
        env = RecordVideo(
            env, 
            video_folder="./training_videos", 
            episode_trigger=record_trigger,
            name_prefix="eval_run"
        )

    obs, info = env.reset()
    agent_prev_info = []
    agent_all_info = []
    episode_return = 0
    episode_returns = []
    episode_length = 0
    episode_lengths = []

    episode_end_step = []



    if debug:
        it = tqdm.tqdm(range(num_steps))
    else:
        it = range(num_steps)

    for step in it:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        episode_length += 1
        agent_info  = agent.process_transition(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
    
        if agent_info is not None:
            agent_info["step"] = step
            agent_prev_info.append(agent_info)
            agent_all_info.append(agent_info)

        if terminated or truncated:
            obs, info = env.reset()
            episode_returns.append(episode_return)
            episode_return = 0

            episode_lengths.append(episode_length)
            episode_length = 0

            episode_end_step.append(step)
    
            if debug and len(episode_returns) % 100 == 0:
                ep_returns = episode_returns[-100 :]
                print(f"episode{len(episode_returns)} , step {step}: last 100 episode return mean: {np.mean(ep_returns)}, min: {np.min(ep_returns)} max: {np.max(ep_returns)}, average episode length: {np.mean(episode_lengths[-100:])}")
                agent.print_info(agent_prev_info)
                agent_prev_info = []
        
    return episode_returns, episode_end_step, agent_all_info
