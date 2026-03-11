
import gymnasium
import numpy as np
import torch 
import numpy as np

class PPPOAgent:

    def __init__(self, actor_net, critic_net, buffer,
                gamma = 0.99, 
                gae_lambda = 0.95,
                ppo_clip = 0.2,
                ppo_early_stopping_kl = 0.01,
                entropy_bonus = 0.001,

                actor_alpha = 1e-4,
                critic_alpha= 1e-4,

                actor_final_alpha = 1e-5,
                critic_final_alpha= 1e-5,
                falloff_steps = 450,
                transitions_per_gradient_update = 2048,
                minibatch_adv_normalization = True,
                use_updated_critic_adv = False,
                critic_loss_clip = None,


                critic_batch_size = 200,
                critic_update_iters=10,
                actor_batch_size = 200,
                actor_update_iters=1,
              
                input_preprocessor = (lambda x : x),
                reward_phi = (lambda x : x),

                # PPPO HYPERS:
            
                actor_perturb_steps = 30,
                actor_perturb_sigma = 1e-4,
        
                 ):

        self.actor = actor_net.to(input_preprocessor.device)
        self.critic = critic_net

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer= buffer
        self.ppo_clip = ppo_clip
        self.ppo_early_stopping_kl = ppo_early_stopping_kl
        self.entropy_bonus = entropy_bonus

        # Actor and critic optmizers
        self.actor_optim = torch.optim.Adam(actor_net.parameters(), actor_alpha, eps=1e-5) 
        self.critic_optim = torch.optim.Adam(critic_net.parameters(), critic_alpha, eps=1e-5) 
        # Actor and critic decay rate
        actor_decay_rate = np.pow(actor_final_alpha / actor_alpha, 1/falloff_steps)
        critic_decay_rate = np.pow(critic_final_alpha /critic_alpha, 1/falloff_steps)
        self.actor_sched = torch.optim.lr_scheduler.LambdaLR(self.actor_optim, lambda step : max(actor_decay_rate**step, actor_final_alpha/actor_alpha) )
        self.critic_sched = torch.optim.lr_scheduler.LambdaLR(self.critic_optim, lambda step : max(critic_decay_rate**step, critic_final_alpha/critic_alpha) )
        # actor and critic settings
        self.critic_loss_clip = critic_loss_clip
        self.minibatch_adv_normalization = minibatch_adv_normalization
        self.use_updated_critic_advantages = use_updated_critic_adv
        self.critic_batch_size = critic_batch_size
        self.critic_update_passes= critic_update_iters
        self.actor_update_passes = actor_update_iters
        self.actor_batch_size = actor_batch_size

        self.transitions_per_gradient_update = transitions_per_gradient_update
        # input preprocesser must work on (obs_shape) and (batch_size, obs_shape)
        self.input_preprocessor = input_preprocessor
        self.reward_phi = reward_phi
        self.last_logp = None

        # PPPO specific hypers
        self.actor_perturb_steps = actor_perturb_steps
        self.actor_perturb_noise_std = actor_perturb_sigma

    def compute_gae(self, states, rewards, next_states, terminated, truncated, lam=0.95):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            # 1. Calculate TD-Errors (delta)
            # delta = r + gamma * V(s') - V(s)
            deltas = rewards + self.gamma * next_values * (1 - terminated) - values

            # 2. Recursive GAE calculation (Backwards through the batch)
            advantages = torch.zeros_like(rewards)
            last_gae = 0

            for t in reversed(range(len(deltas))):
                # Combined discount: gamma * lambda
                if terminated[t] == 1.0 or truncated[t] == 1.0:
                    last_gae = 0
                last_gae = deltas[t] + self.gamma * lam * last_gae
                advantages[t] = last_gae
            # 3. Targets for the Critic (Value = Advantage + current V-estimate)
            targets = advantages + values
        return advantages, targets

    def gradient_update(self):
        buffer_length = len(self.buffer)
        states, actions, rewards, next_states, terminated, truncated = self.buffer.flush()
        states = self.input_preprocessor.process_batch(states)
        next_states = self.input_preprocessor.process_batch(next_states)
        info = {}

        with torch.no_grad():
            advantages, targets = self.compute_gae(states, rewards, next_states, terminated, truncated)
            targets = targets.detach()
            advantages = advantages.detach()

            # normalize advantages at the batch level if we don't do minibatch normalization
            if not self.minibatch_adv_normalization: 
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            explained_var_before_update = 1- torch.var(targets - self.critic(states)) / (torch.var(targets) + 1e-8)
            info["explained_variance_before_update"] = explained_var_before_update.detach().cpu().item()

            old_values = None
            if self.critic_loss_clip is not None:
                old_values = self.critic(states).detach()

        # critic update
        for _ in range(self.critic_update_passes):
            idx = torch.randperm(buffer_length)
            for i in range(0, buffer_length, self.critic_batch_size):
                batch = idx[i:i+self.critic_batch_size]


                predicted = self.critic(states[batch])
                ve = (predicted - targets[batch])

                if self.critic_loss_clip is not None:
                    predicted_clip = old_values[batch] + torch.clamp(predicted - old_values[batch], -self.critic_loss_clip, self.critic_loss_clip)
                    critic_loss = torch.mean(torch.max(
                        (predicted_clip-targets[batch])**2,
                        (predicted - targets[batch])**2
                    ))
                    
                else:
                    ve = predicted - targets[batch]
                    critic_loss = torch.mean(ve**2)


                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
        


        

        # actor update
        with torch.no_grad():
            old_logp = self.actor.compute_log_prob(states, actions)[0].detach()
            if self.use_updated_critic_advantages:
                advantages, targets = self.compute_gae(states, rewards, next_states, terminated, truncated)
                targets = targets.detach()
                advantages = advantages.detach()
                if not self.minibatch_adv_normalization:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
            # perturb actor with kl limit


            for i in range(self.actor_perturb_steps):
                flattened_actor = self.actor.get_params()[:]
                noise = torch.randn_like(flattened_actor) * self.actor_perturb_noise_std
                self.actor.set_params(flattened_actor + noise)
                logp_perturbed = self.actor.compute_log_prob(states, actions)[0].detach()
                log_ratio = logp_perturbed - old_logp
                kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                info["final_perturb_kl"] = kl

                if kl > self.ppo_early_stopping_kl*0.9:
                    print("kl limit reached at perturb step ", i, "backtracking")
                    self.actor.set_params(flattened_actor)
                


        # ppo update with clip 
        for _pass in range(self.actor_update_passes):
            indexes = torch.randperm(buffer_length)
            epoch_kls = []
            for i in range(0, buffer_length, self.actor_batch_size):
                batch = indexes[i:i+self.actor_batch_size]

                if self.minibatch_adv_normalization:
                    adv = (advantages[batch] - advantages[batch].mean()) / (advantages[batch].std() + 1e-8).detach()
                else:
                    adv = advantages[batch].detach()

                logp, entropy, mu, sigma = self.actor.compute_log_prob(states[batch], actions[batch])
                ratio = torch.exp(logp - old_logp[batch])
                clipped_ratio = torch.clamp(ratio, 1-self.ppo_clip, 1+self.ppo_clip)
                ppo_clip_obj = torch.min(ratio * adv, clipped_ratio * adv)
                ppo_loss = -torch.mean(ppo_clip_obj) - self.entropy_bonus*torch.mean(entropy)
                self.actor_optim.zero_grad()
                ppo_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optim.step()

                with torch.no_grad():
                    log_ratio = logp - old_logp[batch]
                    batch_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    epoch_kls.append(batch_kl)
            
            approx_kl = np.mean(epoch_kls)
            info["approx_kl"] = approx_kl
            
            if approx_kl > self.ppo_early_stopping_kl:
                print(f"early stopping {_pass}/{self.actor_update_passes}")
                break
        
        with torch.no_grad():
            batch = torch.randperm(buffer_length)
            critic_error = torch.mean((self.critic(states[batch]) - targets[batch])**2)
            info["critic_loss"] = critic_error.detach().cpu().item()
            logp, entropy, mu, sigma = self.actor.compute_log_prob(states[batch], actions[batch])
            ratio = torch.exp(logp - old_logp[batch])
            clipped_ratio = torch.clamp(ratio, 1-self.ppo_clip, 1+self.ppo_clip)
            ppo_clip_obj = torch.min(ratio * advantages[batch], clipped_ratio * advantages[batch])
            ppo_loss = -torch.mean(ppo_clip_obj) 
            info["actor_loss"] = ppo_loss.detach().cpu().item()
            info["mean_sigma"] = sigma.mean().detach().cpu().item()
            info["min_sigma"] = sigma.min().detach().cpu().item()
            info["max_sigma"] = sigma.max().detach().cpu().item()
            info["min_mu"] = mu.min().detach().cpu().item()
            info["max_mu"] = mu.max().detach().cpu().item()
            info["ratio"] = torch.max(torch.exp(logp - old_logp[batch]))

            info["explained_variance"] = 1- torch.var(targets[batch] - self.critic(states[batch])) / (torch.var(targets[batch]) + 1e-8)

            info["actor_lr"] = self.actor_optim.param_groups[0]["lr"]

        self.actor_sched.step()
        self.critic_sched.step()

        return info

    def print_info(self, agent_info):
        for i in agent_info:
            print(
                "".join([f"{k}: {i[k]}, " for k in i.keys()])

            )



    def act(self, obs):
        with torch.no_grad():
            obs = self.input_preprocessor(obs, update=True)

            obs = torch.as_tensor(obs).squeeze(0)
            action = self.actor.sample_action(obs)
            action = action.detach().cpu().numpy()
            
            return action

    def process_transition(self, obs, act, reward, next_obs, terminated,truncated):
        reward = self.reward_phi(reward, terminated, truncated)

        self.buffer.push(obs, act, reward, next_obs, terminated, truncated)

        if len(self.buffer) >= self.transitions_per_gradient_update:
            self.actor.to("cuda")
            info = self.gradient_update()
            self.actor.to(self.input_preprocessor.device)
            return info