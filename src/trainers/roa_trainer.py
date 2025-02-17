import os
import torch
import numpy as np
from dataclasses import dataclass, asdict
from itertools import count
import time
import random
import statistics
from datetime import datetime
import functools
import wandb

from modelsaver import ModelSaver
from agents.roa_ppo import OnPolicyRunner

class ROATrainer(OnPolicyRunner):

  def __init__(self, 
               env,
               logger, 
               train_config, 
               num_episodes:int,
               init_at_random_ep_len:bool,
               save=True, 
               savedir="run", 
               device="cpu", 
               log_level=1, 
               render=False, 
               group_name="", 
               run_name="run",
               seed=None,
               ):
    """
    Class that trains agents in an environment
    """

    # save input arguments (note the underlying class saves its args too)
    self.logger = logger
    self.num_training_episodes = num_episodes
    self.init_at_random_ep_len = init_at_random_ep_len
    self.device = device
    self.log_level = log_level
    self.render = render
    self.rngseed = seed
    self.run_name = run_name
    self.group_name = group_name

    # initialise underlying on-policy runner (RSL version of trainer)
    super().__init__(env=env.env, train_cfg=train_config, log_dir=savedir, 
                     init_wandb=self.logger.log_to_wandb, device=device)

    if self.log_level > 0:
      print("ROATrainer settings:")
      print(" -> Run name:", self.run_name)
      print(" -> Group name:", self.group_name)
      print(" -> Using device:", self.device)

  def train(self):
    """
    Run the training using the underlying on-policy runner from RSL.
    """

    print(f"ROATrainer.train(): Preparing a training with {self.num_training_episodes} episodes")

    self.learn(num_learning_iterations=self.num_training_episodes,
               init_at_random_ep_len=self.init_at_random_ep_len)
    
    print(f"ROATrainer.train(): Finished a training with {self.num_training_episodes} episodes")

  def log(self, locs, width=80, pad=35):
    """
    Log training output given the logger stored at self.logger
    """

    self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
    self.tot_time += locs['collection_time'] + locs['learn_time']
    iteration_time = locs['collection_time'] + locs['learn_time']

    mean_std = self.alg.actor_critic.std.mean()
    fps = int(self.num_steps_per_env * self.env.num_envs / \
              (locs['collection_time'] + locs['learn_time']))

    self.logger.scalar('Loss/value_function', 'mean_value_loss')
    self.logger.scalar('Loss/surrogate', locs['mean_surrogate_loss'])
    self.logger.scalar('Loss/estimator', locs['mean_estimator_loss'])
    self.logger.scalar('Loss/hist_latent_loss', locs['mean_hist_latent_loss'])
    self.logger.scalar('Loss/priv_reg_loss', locs['mean_priv_reg_loss'])
    self.logger.scalar('Loss/priv_ref_lambda', locs['priv_reg_coef'])
    self.logger.scalar('Loss/entropy_coef', locs['entropy_coef'])
    self.logger.scalar('Loss/learning_rate', self.alg.learning_rate)
    self.logger.scalar('Loss/discriminator', locs['mean_disc_loss'])
    self.logger.scalar('Loss/discriminator_accuracy', locs['mean_disc_acc'])
    self.logger.scalar('Policy/mean_noise_std', mean_std.item())
    self.logger.scalar('Perf/total_fps', fps)
    self.logger.scalar('Perf/collection_time', locs['collection_time'])
    self.logger.scalar('Perf/learning_time', locs['learn_time'])

    if len(locs['rewbuffer']) > 0:
      self.logger.scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']))
      self.logger.scalar('Train/mean_reward_explr', statistics.mean(locs['rew_explr_buffer']))
      self.logger.scalar('Train/mean_reward_task', statistics.mean(locs['rewbuffer'])
                                                   - statistics.mean(locs['rew_explr_buffer']))
      self.logger.scalar('Train/mean_reward_entropy', statistics.mean(locs['rew_entropy_buffer']))
      self.logger.scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']))

    if locs['ep_infos']:
      for key in locs['ep_infos'][0]:
        infotensor = torch.tensor([], device=self.device)
        for ep_info in locs['ep_infos']:
          # handle scalar and zero dimensional tensor infos
          if not isinstance(ep_info[key], torch.Tensor):
            ep_info[key] = torch.Tensor([ep_info[key]])
          if len(ep_info[key].shape) == 0:
            ep_info[key] = ep_info[key].unsqueeze(0)
          infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        value = torch.mean(infotensor).cpu().numpy()

        self.logger.scalar("Episode_reward/" + key, value)

    curr_it = locs['it'] - self.start_learning_iteration
    eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
    mins = eta // 60
    secs = eta % 60
    log_string = (f""""""
                  f"""{'Learning iteration:':>{pad} {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']}}\n"""
                  f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                  f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                  f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                  f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
    
    self.logger.log_step(print_string=log_string)