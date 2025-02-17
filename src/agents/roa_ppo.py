# for VecEnv
from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union

# for on policy runner
import wandb
import time
import os
from collections import deque
import statistics
from copy import copy, deepcopy
import warnings

# for base config
import inspect

# general use
import code
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU

# ----- rsl_rl.utils.utils ----- #

import torch

def split_and_pad_trajectories(tensor, dones):
  """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
  Returns masks corresponding to valid parts of the trajectories
  Example: 
      Input: [ [a1, a2, a3, a4 | a5, a6],
               [b1, b2 | b3, b4, b5 | b6]
              ]

      Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
               [a5, a6, 0, 0],   |    [True, True, False, False],
               [b1, b2, 0, 0],   |    [True, True, False, False],
               [b3, b4, b5, 0],  |    [True, True, True, False],
               [b6, 0, 0, 0]     |    [True, False, False, False],
              ]                  | ]    

  Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
  """
  dones = dones.clone()
  dones[-1] = 1
  # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
  flat_dones = dones.transpose(1, 0).reshape(-1, 1)

  # Get length of trajectory by counting the number of successive not done elements
  done_indices = torch.cat((flat_dones.new_tensor(
      [-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
  trajectory_lengths = done_indices[1:] - done_indices[:-1]
  trajectory_lengths_list = trajectory_lengths.tolist()
  # Extract the individual trajectories
  trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
  padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


  trajectory_masks = trajectory_lengths > torch.arange(
      0, tensor.shape[0], device=tensor.device).unsqueeze(1)
  return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
  """ Does the inverse operation of  split_and_pad_trajectories()
  """
  # Need to transpose before and after the masking to have proper reshaping
  return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)

# ----- rsl_rl.storage.rollout_storage ----- #

class RolloutStorage:

  class Transition:
    def __init__(self):
      self.observations = None
      self.critic_observations = None
      self.actions = None
      self.rewards = None
      self.dones = None
      self.values = None
      self.actions_log_prob = None
      self.action_mean = None
      self.action_sigma = None
      self.hidden_states = None
    def clear(self):
      self.__init__()

  def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):

    self.device = device

    self.obs_shape = obs_shape
    self.privileged_obs_shape = privileged_obs_shape
    self.actions_shape = actions_shape

    # Core
    self.observations = torch.zeros(
        num_transitions_per_env, num_envs, *obs_shape, device=self.device)

    if privileged_obs_shape[0] is not None:
      self.privileged_observations = torch.zeros(
          num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
    else:
      self.privileged_observations = None
    self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    self.actions = torch.zeros(num_transitions_per_env, num_envs,
                               *actions_shape, device=self.device)
    self.dones = torch.zeros(num_transitions_per_env, num_envs,
                             1, device=self.device).byte()

    # For PPO
    self.actions_log_prob = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)
    self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
    self.advantages = torch.zeros(num_transitions_per_env,
                                  num_envs, 1, device=self.device)
    self.mu = torch.zeros(num_transitions_per_env, num_envs,
                          *actions_shape, device=self.device)
    self.sigma = torch.zeros(num_transitions_per_env, num_envs,
                             *actions_shape, device=self.device)

    self.num_transitions_per_env = num_transitions_per_env
    self.num_envs = num_envs

    # rnn
    self.saved_hidden_states_a = None
    self.saved_hidden_states_c = None

    self.step = 0

  def add_transitions(self, transition: Transition):
    if self.step >= self.num_transitions_per_env:
      raise AssertionError("Rollout buffer overflow")
    self.observations[self.step].copy_(transition.observations)
    if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(
        transition.critic_observations)
    self.actions[self.step].copy_(transition.actions)
    self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.values[self.step].copy_(transition.values)
    self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
    self.mu[self.step].copy_(transition.action_mean)
    self.sigma[self.step].copy_(transition.action_sigma)

    self._save_hidden_states(transition.hidden_states)
    self.step += 1

  def _save_hidden_states(self, hidden_states):
    if hidden_states is None or hidden_states==(None, None):
      return
    # make a tuple out of GRU hidden state sto match the LSTM format
    hid_a = hidden_states[0] if isinstance(
        hidden_states[0], tuple) else (hidden_states[0],)
    hid_c = hidden_states[1] if isinstance(
        hidden_states[1], tuple) else (hidden_states[1],)

    # initialize if needed 
    if self.saved_hidden_states_a is None:
      self.saved_hidden_states_a = [torch.zeros(
          self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
      self.saved_hidden_states_c = [torch.zeros(
          self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
    # copy the states
    for i in range(len(hid_a)):
      self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
      self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

  def clear(self):
    self.step = 0

  def compute_returns(self, last_values, gamma, lam):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):
      if step == self.num_transitions_per_env - 1:
        next_values = last_values
      else:
        next_values = self.values[step + 1]
      next_is_not_terminal = 1.0 - self.dones[step].float()
      delta = self.rewards[step] + next_is_not_terminal * \
          gamma * next_values - self.values[step]
      advantage = delta + next_is_not_terminal * gamma * lam * advantage
      self.returns[step] = advantage + self.values[step]

    # Compute and normalize the advantages
    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()) / \
        (self.advantages.std() + 1e-8)

  def get_statistics(self):
    done = self.dones
    done[-1] = 1
    flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
    done_indices = torch.cat((flat_dones.new_tensor(
        [-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
    trajectory_lengths = (done_indices[1:] - done_indices[:-1])
    return trajectory_lengths.float().mean(), self.rewards.mean()

  def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches*mini_batch_size,
                             requires_grad=False, device=self.device)

    observations = self.observations.flatten(0, 1)

    # # shift the observations by one step to the left to get the next observations
    # next_disc_observations = torch.cat((self.disc_observations[1:], self.disc_observations[-1].unsqueeze(0)), dim=0)
    # done_indices = self.dones.nonzero(as_tuple=False).squeeze()
    # next_disc_observations[done_indices] = self.disc_observations[done_indices]
    # next_disc_observations = next_disc_observations.flatten(0, 1)

    if self.privileged_observations is not None:
      critic_observations = self.privileged_observations.flatten(0, 1)
    else:
      critic_observations = observations

    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_mu = self.mu.flatten(0, 1)
    old_sigma = self.sigma.flatten(0, 1)

    for epoch in range(num_epochs):
      for i in range(num_mini_batches):

        start = i*mini_batch_size
        end = (i+1)*mini_batch_size
        batch_idx = indices[start:end]

        obs_batch = observations[batch_idx]
        critic_observations_batch = critic_observations[batch_idx]
        actions_batch = actions[batch_idx]
        target_values_batch = values[batch_idx]
        returns_batch = returns[batch_idx]
        old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
        advantages_batch = advantages[batch_idx]
        old_mu_batch = old_mu[batch_idx]
        old_sigma_batch = old_sigma[batch_idx]


        yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None

       # for RNNs only

  def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

    padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
        self.observations, self.dones)
    if self.privileged_observations is not None: 
      padded_critic_obs_trajectories, _ = split_and_pad_trajectories(
          self.privileged_observations, self.dones)
    else: 
      padded_critic_obs_trajectories = padded_obs_trajectories

    mini_batch_size = self.num_envs // num_mini_batches
    for ep in range(num_epochs):
      first_traj = 0
      for i in range(num_mini_batches):
        start = i*mini_batch_size
        stop = (i+1)*mini_batch_size

        dones = self.dones.squeeze(-1)
        last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        last_was_done[1:] = dones[:-1]
        last_was_done[0] = True
        trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
        last_traj = first_traj + trajectories_batch_size

        masks_batch = trajectory_masks[:, first_traj:last_traj]
        obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
        critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

        actions_batch = self.actions[:, start:stop]
        old_mu_batch = self.mu[:, start:stop]
        old_sigma_batch = self.sigma[:, start:stop]
        returns_batch = self.returns[:, start:stop]
        advantages_batch = self.advantages[:, start:stop]
        values_batch = self.values[:, start:stop]
        old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

        # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
        # then take only time steps after dones (flattens num envs and time dimensions),
        # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
        last_was_done = last_was_done.permute(1, 0)
        hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                        for saved_hidden_states in self.saved_hidden_states_a ] 
        hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                        for saved_hidden_states in self.saved_hidden_states_c ]
        # remove the tuple for GRU
        hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
        hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

        yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                hid_a_batch, hid_c_batch), masks_batch

        first_traj = last_traj

# ----- rsl_rl.env.vec_env ----- #

# minimal interface of the environment
class VecEnv(ABC):
  num_envs: int
  num_obs: int
  num_privileged_obs: int
  num_actions: int
  max_episode_length: int
  privileged_obs_buf: torch.Tensor
  obs_buf: torch.Tensor 
  rew_buf: torch.Tensor
  reset_buf: torch.Tensor
  episode_length_buf: torch.Tensor # current episode duration
  extras: dict
  device: torch.device
  @abstractmethod
  def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
    pass
  @abstractmethod
  def reset(self, env_ids: Union[list, torch.Tensor]):
    pass
  @abstractmethod
  def get_observations(self) -> torch.Tensor:
    pass
  @abstractmethod
  def get_privileged_observations(self) -> Union[torch.Tensor, None]:
    pass

# ----- rsl_rl.modules.actor_critic ----- #

class StateHistoryEncoder(nn.Module):

  def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
    # self.device = device
    super(StateHistoryEncoder, self).__init__()
    self.activation_fn = activation_fn
    self.tsteps = tsteps

    channel_size = 10
    # last_activation = nn.ELU()

    self.encoder = nn.Sequential(
        nn.Linear(input_size, 3 * channel_size), self.activation_fn,
    )

    if tsteps == 50:
      self.conv_layers = nn.Sequential(
          nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * \
                    channel_size, kernel_size = 8, stride = 4), self.activation_fn,
          nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size,
                    kernel_size = 5, stride = 1), self.activation_fn,
          nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
    elif tsteps == 10:
      self.conv_layers = nn.Sequential(
          nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * \
                    channel_size, kernel_size = 4, stride = 2), self.activation_fn,
          nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size,
                    kernel_size = 2, stride = 1), self.activation_fn,
          nn.Flatten())
    elif tsteps == 20:
      self.conv_layers = nn.Sequential(
          nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * \
                    channel_size, kernel_size = 6, stride = 2), self.activation_fn,
          nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size,
                    kernel_size = 4, stride = 2), self.activation_fn,
          nn.Flatten())
    else:
      raise(ValueError("tsteps must be 10, 20 or 50"))

    self.linear_output = nn.Sequential(
        nn.Linear(channel_size * 3, output_size), self.activation_fn
    )

  def forward(self, obs):
    # nd * T * n_proprio
    nd = obs.shape[0]
    T = self.tsteps
    # print("obs device", obs.device)
    # print("encoder device", next(self.encoder.parameters()).device)
    # do projection for n_proprio -> 32
    projection = self.encoder(obs.reshape([nd * T, -1]))
    output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
    output = self.linear_output(output)
    return output

class Actor(nn.Module):

  def __init__(self, num_prop, 
               num_scan, 
               num_actions, 
               scan_encoder_dims,
               actor_hidden_dims, 
               priv_encoder_dims, 
               num_priv_latent, 
               num_priv_explicit, 
               num_hist, activation, 
               tanh_encoder_output=False) -> None:
    super().__init__()
    # prop -> scan -> priv_explicit -> priv_latent -> hist
    # actor input: prop -> scan -> priv_explicit -> latent
    self.num_prop = num_prop
    self.num_scan = num_scan
    self.num_hist = num_hist
    self.num_actions = num_actions
    self.num_priv_latent = num_priv_latent
    self.num_priv_explicit = num_priv_explicit
    self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

    if len(priv_encoder_dims) > 0:
      priv_encoder_layers = []
      priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
      priv_encoder_layers.append(activation)
      for l in range(len(priv_encoder_dims) - 1):
        priv_encoder_layers.append(
            nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
        priv_encoder_layers.append(activation)
      self.priv_encoder = nn.Sequential(*priv_encoder_layers)
      priv_encoder_output_dim = priv_encoder_dims[-1]
    else:
      self.priv_encoder = nn.Identity()
      priv_encoder_output_dim = num_priv_latent

    self.history_encoder = StateHistoryEncoder(
        activation, num_prop, num_hist, priv_encoder_output_dim)

    if self.if_scan_encode:
      scan_encoder = []
      scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
      scan_encoder.append(activation)
      for l in range(len(scan_encoder_dims) - 1):
        if l == len(scan_encoder_dims) - 2:
          scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l+1]))
          scan_encoder.append(nn.Tanh())
        else:
          scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
          scan_encoder.append(activation)
      self.scan_encoder = nn.Sequential(*scan_encoder)
      self.scan_encoder_output_dim = scan_encoder_dims[-1]
    else:
      self.scan_encoder = nn.Identity()
      self.scan_encoder_output_dim = num_scan

    actor_layers = []
    actor_layers.append(nn.Linear(num_prop+
                                  self.scan_encoder_output_dim+
                                  num_priv_explicit+
                                  priv_encoder_output_dim, 
                                  actor_hidden_dims[0]))
    actor_layers.append(activation)
    for l in range(len(actor_hidden_dims)):
      if l == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
      else:
        actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
        actor_layers.append(activation)
    if tanh_encoder_output:
      actor_layers.append(nn.Tanh())
    self.actor_backbone = nn.Sequential(*actor_layers)

  def forward(self, obs, hist_encoding: bool, eval=False, scandots_latent=None):
    if not eval:
      if self.if_scan_encode:
        obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        if scandots_latent is None:
          scan_latent = self.scan_encoder(obs_scan)   
        else:
          scan_latent = scandots_latent
        obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
      else:
        obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
      obs_priv_explicit = obs[:, self.num_prop + \
                              self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
      if hist_encoding:
        latent = self.infer_hist_latent(obs)
      else:
        latent = self.infer_priv_latent(obs)
      backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
      backbone_output = self.actor_backbone(backbone_input)
      return backbone_output
    else:
      if self.if_scan_encode:
        obs_scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        if scandots_latent is None:
          scan_latent = self.scan_encoder(obs_scan)   
        else:
          scan_latent = scandots_latent
        obs_prop_scan = torch.cat([obs[:, :self.num_prop], scan_latent], dim=1)
      else:
        obs_prop_scan = obs[:, :self.num_prop + self.num_scan]
      obs_priv_explicit = obs[:, self.num_prop + \
                              self.num_scan:self.num_prop + self.num_scan + self.num_priv_explicit]
      if hist_encoding:
        latent = self.infer_hist_latent(obs)
      else:
        latent = self.infer_priv_latent(obs)
      backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
      backbone_output = self.actor_backbone(backbone_input)
      return backbone_output

  def infer_priv_latent(self, obs):
    priv = obs[:, self.num_prop + self.num_scan + self.num_priv_explicit: self.num_prop + \
               self.num_scan + self.num_priv_explicit + self.num_priv_latent]
    return self.priv_encoder(priv)

  def infer_hist_latent(self, obs):
    hist = obs[:, -self.num_hist*self.num_prop:]
    return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))

  def infer_scandots_latent(self, obs):
    scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
    return self.scan_encoder(scan)

class ActorCriticRMA(nn.Module):

  is_recurrent = False

  def __init__(self,  num_prop,
               num_scan,
               num_critic_obs,
               num_priv_latent, 
               num_priv_explicit,
               num_hist,
               num_actions,
               scan_encoder_dims=[256, 256, 256],
               actor_hidden_dims=[256, 256, 256],
               critic_hidden_dims=[256, 256, 256],
               activation='elu',
               init_noise_std=1.0,
               **kwargs):
    if kwargs:
      print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + \
            str([key for key in kwargs.keys()]))
    super(ActorCriticRMA, self).__init__()

    self.kwargs = kwargs
    priv_encoder_dims= kwargs['priv_encoder_dims']
    activation = get_activation(activation)

    self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims,
                       num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=kwargs['tanh_encoder_output'])


    # Value function
    critic_layers = []
    critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
    critic_layers.append(activation)
    for l in range(len(critic_hidden_dims)):
      if l == len(critic_hidden_dims) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
      else:
        critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
        critic_layers.append(activation)
    self.critic = nn.Sequential(*critic_layers)

    # Action noise
    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    self.distribution = None
    # disable args validation for speedup
    Normal.set_default_validate_args = False

    # seems that we get better performance without init
    # self.init_memory_weights(self.memory_a, 0.001, 0.)
    # self.init_memory_weights(self.memory_c, 0.001, 0.)

  @staticmethod
  # not used at the moment
  def init_weights(sequential, scales):
    [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
     enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

  def reset(self, dones=None):
    pass

  def forward(self):
    raise NotImplementedError

  @property
  def action_mean(self):
    return self.distribution.mean

  @property
  def action_std(self):
    return self.distribution.stddev

  @property
  def entropy(self):
    return self.distribution.entropy().sum(dim=-1)

  def update_distribution(self, observations, hist_encoding):
    mean = self.actor(observations, hist_encoding)
    self.distribution = Normal(mean, mean*0. + self.std)

  def act(self, observations, hist_encoding=False, **kwargs):
    self.update_distribution(observations, hist_encoding)
    return self.distribution.sample()

  def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)

  def act_inference(self, observations, hist_encoding=False, eval=False, scandots_latent=None, **kwargs):
    if not eval:
      actions_mean = self.actor(observations, hist_encoding, eval, scandots_latent)
      return actions_mean
    else:
      actions_mean, latent_hist, latent_priv = self.actor(
          observations, hist_encoding, eval=True)
      return actions_mean, latent_hist, latent_priv

  def evaluate(self, critic_observations, **kwargs):
    value = self.critic(critic_observations)
    return value

  def reset_std(self, std, num_actions, device):
    new_std = std * torch.ones(num_actions, device=device)
    self.std.data = new_std.data

def get_activation(act_name):
  if act_name == "elu":
    return nn.ELU()
  elif act_name == "selu":
    return nn.SELU()
  elif act_name == "relu":
    return nn.ReLU()
  elif act_name == "crelu":
    return nn.ReLU()
  elif act_name == "lrelu":
    return nn.LeakyReLU()
  elif act_name == "tanh":
    return nn.Tanh()
  elif act_name == "sigmoid":
    return nn.Sigmoid()
  else:
    print("invalid activation function!")
    return None

# ----- rsl_rl.modules.estimator ----- #

class Estimator(nn.Module):
  def __init__(self,  input_dim,
               output_dim,
               hidden_dims=[256, 128, 64],
               activation="elu",
               **kwargs):
    super(Estimator, self).__init__()

    self.input_dim = input_dim
    self.output_dim = output_dim
    activation = get_activation(activation)
    estimator_layers = []
    estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
    estimator_layers.append(activation)
    for l in range(len(hidden_dims)):
      if l == len(hidden_dims) - 1:
        estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
      else:
        estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        estimator_layers.append(activation)
    # estimator_layers.append(nn.Tanh())
    self.estimator = nn.Sequential(*estimator_layers)

  def forward(self, input):
    return self.estimator(input)

  def inference(self, input):
    with torch.no_grad():
      return self.estimator(input)

# ----- rsl_rl.modules.depth_backbone ----- #

class DepthOnlyFCBackbone58x87(nn.Module):
  def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
    super().__init__()

    self.num_frames = num_frames
    activation = nn.ELU()
    self.image_compression = nn.Sequential(
        # [1, 58, 87]
        nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
        # [32, 54, 83]
        nn.MaxPool2d(kernel_size=2, stride=2),
        # [32, 27, 41]
        activation,
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        activation,
        nn.Flatten(),
        # [32, 25, 39]
        nn.Linear(64 * 25 * 39, 128),
        activation,
        nn.Linear(128, scandots_output_dim)
    )

    if output_activation == "tanh":
      self.output_activation = nn.Tanh()
    else:
      self.output_activation = activation

  def forward(self, images: torch.Tensor):
    images_compressed = self.image_compression(images.unsqueeze(1))
    latent = self.output_activation(images_compressed)

    return latent

class RecurrentDepthBackbone(nn.Module):
  def __init__(self, base_backbone, env_cfg) -> None:
    super().__init__()
    activation = nn.ELU()
    last_activation = nn.Tanh()
    self.base_backbone = base_backbone
    if env_cfg == None:
      self.combination_mlp = nn.Sequential(
          nn.Linear(32 + 53, 128),
          activation,
          nn.Linear(128, 32)
      )
    else:
      self.combination_mlp = nn.Sequential(
          nn.Linear(32 + env_cfg.env.n_proprio, 128),
          activation,
          nn.Linear(128, 32)
      )
    self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
    self.output_mlp = nn.Sequential(
        nn.Linear(512, 32+2),
        last_activation
    )
    self.hidden_states = None

  def forward(self, depth_image, proprioception):
    depth_image = self.base_backbone(depth_image)
    depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
    # depth_latent = self.base_backbone(depth_image)
    depth_latent, self.hidden_states = self.rnn(
        depth_latent[:, None, :], self.hidden_states)
    depth_latent = self.output_mlp(depth_latent.squeeze(1))

    return depth_latent

  def detach_hidden_states(self):
    self.hidden_states = self.hidden_states.detach().clone()

# ----- rsl_rl.algorithms.ppo ----- #

class RMS(object):

  def __init__(self, device, epsilon=1e-4, shape=(1,)):
    self.M = torch.zeros(shape, device=device)
    self.S = torch.ones(shape, device=device)
    self.n = epsilon

  def __call__(self, x):
    bs = x.size(0)
    delta = torch.mean(x, dim=0) - self.M
    new_M = self.M + delta * bs / (self.n + bs)
    new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2)
             * self.n * bs / (self.n + bs)) / (self.n + bs)

    self.M = new_M
    self.S = new_S
    self.n += bs

    return self.M, self.S

class PPO:

  actor_critic: ActorCriticRMA

  def __init__(self,
               actor_critic,
               estimator,
               estimator_paras,
               depth_encoder,
               depth_encoder_paras,
               depth_actor,
               num_learning_epochs=1,
               num_mini_batches=1,
               clip_param=0.2,
               gamma=0.998,
               lam=0.95,
               value_loss_coef=1.0,
               entropy_coef=0.0,
               learning_rate=1e-3,
               max_grad_norm=1.0,
               use_clipped_value_loss=True,
               schedule="fixed",
               desired_kl=0.01,
               device='cpu',
               dagger_update_freq=20,
               priv_reg_coef_schedual = [0, 0, 0],
               **kwargs
               ):


    self.device = device

    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate

    # PPO components
    self.actor_critic = actor_critic
    self.actor_critic.to(self.device)
    self.storage = None # initialized later
    self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    self.transition = RolloutStorage.Transition()

    # PPO parameters
    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    # Adaptation
    self.hist_encoder_optimizer = optim.Adam(
        self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
    self.priv_reg_coef_schedual = priv_reg_coef_schedual
    self.counter = 0

    # Estimator
    self.estimator = estimator
    self.priv_states_dim = estimator_paras["priv_states_dim"]
    self.num_prop = estimator_paras["num_prop"]
    self.num_scan = estimator_paras["num_scan"]
    self.estimator_optimizer = optim.Adam(
        self.estimator.parameters(), lr=estimator_paras["learning_rate"])
    self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

    # Depth encoder
    self.if_depth = depth_encoder != None
    if self.if_depth:
      self.depth_encoder = depth_encoder
      self.depth_encoder_optimizer = optim.Adam(
          self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"])
      self.depth_encoder_paras = depth_encoder_paras
      self.depth_actor = depth_actor
      self.depth_actor_optimizer = optim.Adam(
          [*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])

  def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
    self.storage = RolloutStorage(num_envs, num_transitions_per_env,
                                  actor_obs_shape,  critic_obs_shape, action_shape, self.device)

  def test_mode(self):
    self.actor_critic.test()

  def train_mode(self):
    self.actor_critic.train()

  def act(self, obs, critic_obs, info, hist_encoding=False):
    if self.actor_critic.is_recurrent:
      self.transition.hidden_states = self.actor_critic.get_hidden_states()
    # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
    if self.train_with_estimated_states:
      obs_est = obs.clone()
      priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])
      obs_est[:, self.num_prop+self.num_scan:self.num_prop+ \
              self.num_scan+self.priv_states_dim] = priv_states_estimated
      self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
    else:
      self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

    self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
        self.transition.actions).detach()
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    self.transition.observations = obs
    self.transition.critic_observations = critic_obs

    return self.transition.actions

  def process_env_step(self, rewards, dones, infos):
    rewards_total = rewards.clone()

    self.transition.rewards = rewards_total.clone()
    self.transition.dones = dones
    # Bootstrapping on time outs
    if 'time_outs' in infos:
      self.transition.rewards += self.gamma * \
          torch.squeeze(self.transition.values * \
                        infos['time_outs'].unsqueeze(1).to(self.device), 1)

    # Record the transition
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.actor_critic.reset(dones)

    return rewards_total

  def compute_returns(self, last_critic_obs):
    last_values= self.actor_critic.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)

  def update(self):
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_estimator_loss = 0
    mean_discriminator_loss = 0
    mean_discriminator_acc = 0
    mean_priv_reg_loss = 0
    if self.actor_critic.is_recurrent:
      generator = self.storage.reccurent_mini_batch_generator(
          self.num_mini_batches, self.num_learning_epochs)
    else:
      generator = self.storage.mini_batch_generator(
          self.num_mini_batches, self.num_learning_epochs)
    for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

      # match distribution dimension
      self.actor_critic.act(obs_batch, masks=masks_batch,
                            hidden_states=hid_states_batch[0])

      actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
      value_batch = self.actor_critic.evaluate(
          critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
      mu_batch = self.actor_critic.action_mean
      sigma_batch = self.actor_critic.action_std
      entropy_batch = self.actor_critic.entropy

      # Adaptation module update
      priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
      with torch.inference_mode():
        hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
      priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()
                       ).norm(p=2, dim=1).mean()
      priv_reg_stage = min(
          max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
      priv_reg_coef = priv_reg_stage * \
          (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]
           ) + self.priv_reg_coef_schedual[0]

      # Estimator
      # obs in batch is with true priv_states
      priv_states_predicted = self.estimator(obs_batch[:, :self.num_prop])
      estimator_loss = (priv_states_predicted - obs_batch[:, self.num_prop+ \
                        self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim]).pow(2).mean()
      self.estimator_optimizer.zero_grad()
      estimator_loss.backward()
      nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
      self.estimator_optimizer.step()

      # KL
      if self.desired_kl != None and self.schedule == 'adaptive':
        with torch.inference_mode():
          kl = torch.sum(
              torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
          kl_mean = torch.mean(kl)

          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate


      # Surrogate loss
      ratio = torch.exp(actions_log_prob_batch - \
                        torch.squeeze(old_actions_log_prob_batch))
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                         1.0 + self.clip_param)
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # Value function loss
      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      loss = surrogate_loss + \
          self.value_loss_coef * value_loss - \
          self.entropy_coef * entropy_batch.mean() + \
          priv_reg_coef * priv_reg_loss
      # loss = self.teacher_alpha * imitation_loss + (1 - self.teacher_alpha) * loss

      # Gradient step
      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_estimator_loss += estimator_loss.item()
      mean_priv_reg_loss += priv_reg_loss.item()
      mean_discriminator_loss += 0
      mean_discriminator_acc += 0

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_estimator_loss /= num_updates
    mean_priv_reg_loss /= num_updates
    mean_discriminator_loss /= num_updates
    mean_discriminator_acc /= num_updates
    self.storage.clear()
    self.update_counter()
    return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef

  def update_dagger(self):
    mean_hist_latent_loss = 0
    if self.actor_critic.is_recurrent:
      generator = self.storage.reccurent_mini_batch_generator(
          self.num_mini_batches, self.num_learning_epochs)
    else:
      generator = self.storage.mini_batch_generator(
          self.num_mini_batches, self.num_learning_epochs)
    for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
      with torch.inference_mode():
        self.actor_critic.act(obs_batch, hist_encoding=True,
                              masks=masks_batch, hidden_states=hid_states_batch[0])

      # Adaptation module update
      with torch.inference_mode():
        priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
      hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
      hist_latent_loss = (priv_latent_batch.detach() - \
                          hist_latent_batch).norm(p=2, dim=1).mean()
      self.hist_encoder_optimizer.zero_grad()
      hist_latent_loss.backward()
      nn.utils.clip_grad_norm_(
          self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
      self.hist_encoder_optimizer.step()

      mean_hist_latent_loss += hist_latent_loss.item()
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_hist_latent_loss /= num_updates
    self.storage.clear()
    self.update_counter()
    return mean_hist_latent_loss

  def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
    # Depth encoder ditillation
    if self.if_depth:
      # TODO: needs to save hidden states
      depth_encoder_loss = (scandots_latent_batch.detach() - \
                            depth_latent_batch).norm(p=2, dim=1).mean()

      self.depth_encoder_optimizer.zero_grad()
      depth_encoder_loss.backward()
      nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
      self.depth_encoder_optimizer.step()
      return depth_encoder_loss.item()

  def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch):
    if self.if_depth:
      depth_actor_loss = (actions_teacher_batch.detach() - \
                          actions_student_batch).norm(p=2, dim=1).mean()
      yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()

      loss = depth_actor_loss + yaw_loss

      self.depth_actor_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
      self.depth_actor_optimizer.step()
      return depth_actor_loss.item(), yaw_loss.item()

  def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
    if self.if_depth:
      depth_encoder_loss = (scandots_latent_batch.detach() - \
                            depth_latent_batch).norm(p=2, dim=1).mean()
      depth_actor_loss = (actions_teacher_batch.detach() - \
                          actions_student_batch).norm(p=2, dim=1).mean()

      depth_loss = depth_encoder_loss + depth_actor_loss

      self.depth_actor_optimizer.zero_grad()
      depth_loss.backward()
      nn.utils.clip_grad_norm_([*self.depth_actor.parameters(),
                               *self.depth_encoder.parameters()], self.max_grad_norm)
      self.depth_actor_optimizer.step()
      return depth_encoder_loss.item(), depth_actor_loss.item()

  def update_counter(self):
    self.counter += 1

  def compute_apt_reward(self, source, target):

    b1, b2 = source.size(0), target.size(0)
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    # sim_matrix = torch.norm(source[:, None, ::2].view(b1, 1, -1) - target[None, :, ::2].view(1, b2, -1), dim=-1, p=2)
    # sim_matrix = torch.norm(source[:, None, :2].view(b1, 1, -1) - target[None, :, :2].view(1, b2, -1), dim=-1, p=2)
    sim_matrix = torch.norm(source[:, None, :].view(
        b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

    reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

    if not self.knn_avg:  # only keep k-th nearest neighbor
      reward = reward[:, -1]
      reward = reward.reshape(-1, 1)  # (b1, 1)
      if self.rms:
        moving_mean, moving_std = self.disc_state_rms(reward)
        reward = reward / moving_std
      reward = torch.clamp(reward - self.knn_clip, 0)  # (b1, )
    else:  # average over all k nearest neighbors
      reward = reward.reshape(-1, 1)  # (b1 * k, 1)
      if self.rms:
        moving_mean, moving_std = self.disc_state_rms(reward)
        reward = reward / moving_std
      reward = torch.clamp(reward - self.knn_clip, 0)
      reward = reward.reshape((b1, self.knn_k))  # (b1, k)
      reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward

# ----- rsl_rl.runners.on_policy_runner ----- #

class OnPolicyRunner:

  def __init__(self,
               env: VecEnv,
               train_cfg,
               log_dir=None,
               init_wandb=True,
               device='cpu', **kwargs):

    self.cfg=train_cfg["runner"]
    self.alg_cfg = train_cfg["algorithm"]
    self.policy_cfg = train_cfg["policy"]
    self.estimator_cfg = train_cfg["estimator"]
    self.depth_encoder_cfg = train_cfg["depth_encoder"]
    self.device = device
    self.env = env

    print("Using MLP and Priviliged Env encoder ActorCritic structure")
    actor_critic: ActorCriticRMA = ActorCriticRMA(self.env.cfg.env.n_proprio,
                                                  self.env.cfg.env.n_scan,
                                                  self.env.num_obs,
                                                  self.env.cfg.env.n_priv_latent,
                                                  self.env.cfg.env.n_priv,
                                                  self.env.cfg.env.history_len,
                                                  self.env.num_actions,
                                                  **self.policy_cfg).to(self.device)
    estimator = Estimator(input_dim=env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv,
                          hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
    # Depth encoder
    self.if_depth = self.depth_encoder_cfg["if_depth"]
    if self.if_depth:
      depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
                                                self.policy_cfg["scan_encoder_dims"][-1], 
                                                self.depth_encoder_cfg["hidden_dims"],
                                                )
      depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
      depth_actor = deepcopy(actor_critic.actor)
    else:
      depth_encoder = None
      depth_actor = None
    # self.depth_encoder = depth_encoder
    # self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=self.depth_encoder_cfg["learning_rate"])
    # self.depth_encoder_paras = self.depth_encoder_cfg
    # self.depth_encoder_criterion = nn.MSELoss()
    # Create algorithm
    alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
    self.alg: PPO = alg_class(actor_critic, 
                              estimator, self.estimator_cfg, 
                              depth_encoder, self.depth_encoder_cfg, depth_actor,
                              device=self.device, **self.alg_cfg)
    self.num_steps_per_env = self.cfg["num_steps_per_env"]
    self.save_interval = self.cfg["save_interval"]
    self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

    self.alg.init_storage(
        self.env.num_envs, 
        self.num_steps_per_env, 
        [self.env.num_obs], 
        [self.env.num_privileged_obs], 
        [self.env.num_actions],
    )

    self.learn = self.learn_RL if not self.if_depth else self.learn_vision

    # Log
    self.log_dir = log_dir
    self.writer = None
    self.tot_timesteps = 0
    self.tot_time = 0
    self.current_learning_iteration = 0

  def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
    mean_value_loss = 0.
    mean_surrogate_loss = 0.
    mean_estimator_loss = 0.
    mean_disc_loss = 0.
    mean_disc_acc = 0.
    mean_hist_latent_loss = 0.
    mean_priv_reg_loss = 0. 
    priv_reg_coef = 0.
    entropy_coef = 0.
    # initialize writer
    # if self.log_dir is not None and self.writer is None:
    #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
          self.env.episode_length_buf, high=int(self.env.max_episode_length))
    obs = self.env.get_observations()
    privileged_obs = self.env.get_privileged_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
    infos = {}
    infos["depth"] = self.env.depth_buffer.clone().to(
        self.device) if self.if_depth else None
    self.alg.actor_critic.train() # switch to train mode (for dropout for example)

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    rew_explr_buffer = deque(maxlen=100)
    rew_entropy_buffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    cur_reward_explr_sum = torch.zeros(
        self.env.num_envs, dtype=torch.float, device=self.device)
    cur_reward_entropy_sum = torch.zeros(
        self.env.num_envs, dtype=torch.float, device=self.device)
    cur_episode_length = torch.zeros(
        self.env.num_envs, dtype=torch.float, device=self.device)

    tot_iter = self.current_learning_iteration + num_learning_iterations
    self.start_learning_iteration = copy(self.current_learning_iteration)

    for it in range(self.current_learning_iteration, tot_iter):
      start = time.time()
      hist_encoding = it % self.dagger_update_freq == 0

      # Rollout
      with torch.inference_mode():
        for i in range(self.num_steps_per_env):
          actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
          # obs has changed to next_obs !! if done obs has been reset
          obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
          critic_obs = privileged_obs if privileged_obs is not None else obs
          obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(
              self.device), rewards.to(self.device), dones.to(self.device)
          total_rew = self.alg.process_env_step(rewards, dones, infos)

          if self.log_dir is not None:
            # Book keeping
            if 'episode' in infos:
              ep_infos.append(infos['episode'])
            cur_reward_sum += total_rew
            cur_reward_explr_sum += 0
            cur_reward_entropy_sum += 0
            cur_episode_length += 1

            new_ids = (dones > 0).nonzero(as_tuple=False)

            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            rew_explr_buffer.extend(
                cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
            rew_entropy_buffer.extend(
                cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

            cur_reward_sum[new_ids] = 0
            cur_reward_explr_sum[new_ids] = 0
            cur_reward_entropy_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        stop = time.time()
        collection_time = stop - start

        # Learning step
        start = stop
        self.alg.compute_returns(critic_obs)

      mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef = self.alg.update()
      if hist_encoding:
        print("Updating dagger...")
        mean_hist_latent_loss = self.alg.update_dagger()

      stop = time.time()
      learn_time = stop - start
      if self.log_dir is not None:
        self.log(locals())
      if it < 2500:
        if it % self.save_interval == 0:
          self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
      elif it < 5000:
        if it % (2*self.save_interval) == 0:
          self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
      else:
        if it % (5*self.save_interval) == 0:
          self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
      ep_infos.clear()

    # self.current_learning_iteration += num_learning_iterations
    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(
        self.current_learning_iteration)))

  def learn_vision(self, num_learning_iterations, init_at_random_ep_len=False):
    tot_iter = self.current_learning_iteration + num_learning_iterations
    self.start_learning_iteration = copy(self.current_learning_iteration)

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
    cur_episode_length = torch.zeros(
        self.env.num_envs, dtype=torch.float, device=self.device)

    obs = self.env.get_observations()
    infos = {}
    infos["depth"] = self.env.depth_buffer.clone().to(
        self.device)[:, -1] if self.if_depth else None
    infos["delta_yaw_ok"] = torch.ones(
        self.env.num_envs, dtype=torch.bool, device=self.device)
    self.alg.depth_encoder.train()
    self.alg.depth_actor.train()

    num_pretrain_iter = 0
    for it in range(self.current_learning_iteration, tot_iter):
      start = time.time()
      depth_latent_buffer = []
      scandots_latent_buffer = []
      actions_teacher_buffer = []
      actions_student_buffer = []
      yaw_buffer_student = []
      yaw_buffer_teacher = []
      delta_yaw_ok_buffer = []
      for i in range(self.depth_encoder_cfg["num_steps_per_env"]):
        if infos["depth"] != None:
          with torch.no_grad():
            scandots_latent = self.alg.actor_critic.actor.infer_scandots_latent(obs)
          scandots_latent_buffer.append(scandots_latent)
          obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
          obs_prop_depth[:, 6:8] = 0
          # clone is crucial to avoid in-place operation
          depth_latent_and_yaw = self.alg.depth_encoder(
              infos["depth"].clone(), obs_prop_depth)

          depth_latent = depth_latent_and_yaw[:, :-2]
          yaw = 1.5*depth_latent_and_yaw[:, -2:]

          depth_latent_buffer.append(depth_latent)
          yaw_buffer_student.append(yaw)
          yaw_buffer_teacher.append(obs[:, 6:8])

        with torch.no_grad():
          actions_teacher = self.alg.actor_critic.act_inference(
              obs, hist_encoding=True, scandots_latent=None)
          actions_teacher_buffer.append(actions_teacher)

        obs_student = obs.clone()
        # obs_student[:, 6:8] = yaw.detach()
        obs_student[infos["delta_yaw_ok"], 6:8] = yaw.detach()[infos["delta_yaw_ok"]]
        delta_yaw_ok_buffer.append(torch.nonzero(
            infos["delta_yaw_ok"]).size(0) / infos["delta_yaw_ok"].numel())
        actions_student = self.alg.depth_actor(
            obs_student, hist_encoding=True, scandots_latent=depth_latent)
        actions_student_buffer.append(actions_student)

        # detach actions before feeding the env
        if it < num_pretrain_iter:
          # obs has changed to next_obs !! if done obs has been reset
          obs, privileged_obs, rewards, dones, infos = self.env.step(
              actions_teacher.detach())
        else:
          # obs has changed to next_obs !! if done obs has been reset
          obs, privileged_obs, rewards, dones, infos = self.env.step(
              actions_student.detach())
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(
            self.device), rewards.to(self.device), dones.to(self.device)

        if self.log_dir is not None:
            # Book keeping
          if 'episode' in infos:
            ep_infos.append(infos['episode'])
          cur_reward_sum += rewards
          cur_episode_length += 1
          new_ids = (dones > 0).nonzero(as_tuple=False)
          rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
          lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
          cur_reward_sum[new_ids] = 0
          cur_episode_length[new_ids] = 0

      stop = time.time()
      collection_time = stop - start
      start = stop

      delta_yaw_ok_percentage = sum(delta_yaw_ok_buffer) / len(delta_yaw_ok_buffer)
      scandots_latent_buffer = torch.cat(scandots_latent_buffer, dim=0)
      depth_latent_buffer = torch.cat(depth_latent_buffer, dim=0)
      depth_encoder_loss = 0
      # depth_encoder_loss = self.alg.update_depth_encoder(depth_latent_buffer, scandots_latent_buffer)

      actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
      actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
      yaw_buffer_student = torch.cat(yaw_buffer_student, dim=0)
      yaw_buffer_teacher = torch.cat(yaw_buffer_teacher, dim=0)
      depth_actor_loss, yaw_loss = self.alg.update_depth_actor(
          actions_student_buffer, actions_teacher_buffer, yaw_buffer_student, yaw_buffer_teacher)

      # depth_encoder_loss, depth_actor_loss = self.alg.update_depth_both(depth_latent_buffer, scandots_latent_buffer, actions_student_buffer, actions_teacher_buffer)
      stop = time.time()
      learn_time = stop - start

      self.alg.depth_encoder.detach_hidden_states()

      if self.log_dir is not None:
        self.log_vision(locals())
      if (it-self.start_learning_iteration < 2500 and it % self.save_interval == 0) or \
         (it-self.start_learning_iteration < 5000 and it % (2*self.save_interval) == 0) or \
         (it-self.start_learning_iteration >= 5000 and it % (5*self.save_interval) == 0):
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
      ep_infos.clear()

  def log_vision(self, locs, width=80, pad=35):
    self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
    self.tot_time += locs['collection_time'] + locs['learn_time']
    iteration_time = locs['collection_time'] + locs['learn_time']

    ep_string = f''
    wandb_dict = {}
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
        value = torch.mean(infotensor)
        wandb_dict['Episode_rew/' + key] = value
        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    mean_std = self.alg.actor_critic.std.mean()
    fps = int(self.num_steps_per_env * self.env.num_envs / \
              (locs['collection_time'] + locs['learn_time']))

    wandb_dict['Loss_depth/delta_yaw_ok_percent'] = locs['delta_yaw_ok_percentage']
    wandb_dict['Loss_depth/depth_encoder'] = locs['depth_encoder_loss']
    wandb_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
    wandb_dict['Loss_depth/yaw'] = locs['yaw_loss']
    wandb_dict['Policy/mean_noise_std'] = mean_std.item()
    wandb_dict['Perf/total_fps'] = fps
    wandb_dict['Perf/collection time'] = locs['collection_time']
    wandb_dict['Perf/learning_time'] = locs['learn_time']
    if len(locs['rewbuffer']) > 0:
      wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
      wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])

    wandb.log(wandb_dict, step=locs['it'])

    str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

    if len(locs['rewbuffer']) > 0:
      log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                    f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                    f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                    f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                    f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                    f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n""")
    else:
      log_string = (f"""{'#' * width}\n""")

    log_string += f"""{'-' * width}\n"""
    log_string += ep_string
    curr_it = locs['it'] - self.start_learning_iteration
    eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
    mins = eta // 60
    secs = eta % 60
    log_string += (f"""{'-' * width}\n"""
                   f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                   f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                   f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                   f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
    print(log_string)

  def log(self, locs, width=80, pad=35):
    self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
    self.tot_time += locs['collection_time'] + locs['learn_time']
    iteration_time = locs['collection_time'] + locs['learn_time']

    ep_string = f''
    wandb_dict = {}
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
        value = torch.mean(infotensor)
        wandb_dict['Episode_rew/' + key] = value
        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    mean_std = self.alg.actor_critic.std.mean()
    fps = int(self.num_steps_per_env * self.env.num_envs / \
              (locs['collection_time'] + locs['learn_time']))

    wandb_dict['Loss/value_function'] = ['mean_value_loss']
    wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
    wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
    wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
    wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
    wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
    wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
    wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
    wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
    wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

    wandb_dict['Policy/mean_noise_std'] = mean_std.item()
    wandb_dict['Perf/total_fps'] = fps
    wandb_dict['Perf/collection time'] = locs['collection_time']
    wandb_dict['Perf/learning_time'] = locs['learn_time']
    if len(locs['rewbuffer']) > 0:
      wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
      wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
      wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - \
          wandb_dict['Train/mean_reward_explr']
      wandb_dict['Train/mean_reward_entropy'] = statistics.mean(
          locs['rew_entropy_buffer'])
      wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
      # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
      # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

    wandb.log(wandb_dict, step=locs['it'])

    str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

    if len(locs['rewbuffer']) > 0:
      log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                    f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                    f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                    f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                    f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                    f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                    f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
      #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
      #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
    else:
      log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                    f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
      #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
      #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

    log_string += f"""{'-' * width}\n"""
    log_string += ep_string
    curr_it = locs['it'] - self.start_learning_iteration
    eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
    mins = eta // 60
    secs = eta % 60
    log_string += (f"""{'-' * width}\n"""
                   f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                   f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                   f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                   f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
    print(log_string)

  def save(self, path, infos=None):
    state_dict = {
        'model_state_dict': self.alg.actor_critic.state_dict(),
        'estimator_state_dict': self.alg.estimator.state_dict(),
        'optimizer_state_dict': self.alg.optimizer.state_dict(),
        'iter': self.current_learning_iteration,
        'infos': infos,
    }
    if self.if_depth:
      state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
      state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
    torch.save(state_dict, path)

  def load(self, path, load_optimizer=True):
    print("*" * 80)
    print("Loading model from {}...".format(path))
    loaded_dict = torch.load(path, map_location=self.device)
    self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
    if self.if_depth:
      if 'depth_encoder_state_dict' not in loaded_dict:
        warnings.warn(
            "'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
      else:
        print("Saved depth encoder detected, loading...")
        self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
      if 'depth_actor_state_dict' in loaded_dict:
        print("Saved depth actor detected, loading...")
        self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
      else:
        print("No saved depth actor, Copying actor critic actor to depth actor...")
        self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
    if load_optimizer:
      self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
    # self.current_learning_iteration = loaded_dict['iter']
    print("*" * 80)
    return loaded_dict['infos']

  def get_inference_policy(self, device=None):
    self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
    if device is not None:
      self.alg.actor_critic.to(device)
    return self.alg.actor_critic.act_inference

  def get_depth_actor_inference_policy(self, device=None):
    self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
    if device is not None:
      self.alg.depth_actor.to(device)
    return self.alg.depth_actor

  def get_actor_critic(self, device=None):
    self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
    if device is not None:
      self.alg.actor_critic.to(device)
    return self.alg.actor_critic

  def get_estimator_inference_policy(self, device=None):
    self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
    if device is not None:
      self.alg.estimator.to(device)
    return self.alg.estimator.inference

  def get_depth_encoder_inference_policy(self, device=None):
    self.alg.depth_encoder.eval()
    if device is not None:
      self.alg.depth_encoder.to(device)
    return self.alg.depth_encoder

  def get_disc_inference_policy(self, device=None):
    self.alg.discriminator.eval() # switch to evaluation mode (dropout for example)
    if device is not None:
      self.alg.discriminator.to(device)
    return self.alg.discriminator.inference

# ----- legged_gym.envs.base.base_config ----- #

class BaseConfig:

  def __init__(self) -> None:
    """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
    self.init_member_classes(self)

  @staticmethod
  def init_member_classes(obj):
    # iterate over all attributes names
    for key in dir(obj):
        # disregard builtin attributes
        # if key.startswith("__"):
      if key=="__class__":
        continue
      # get the corresponding attribute object
      var =  getattr(obj, key)
      # check if it the attribute is a class
      if inspect.isclass(var):
        # instantate the class
        i_var = var()
        # set the attribute to the instance instead of the type
        setattr(obj, key, i_var)
        # recursively init members of the attribute
        BaseConfig.init_member_classes(i_var)

# ----- legged_gym.scripts.legged_gyms.envs.base.legged_robot_config ----- #

class LeggedRobotCfgLocal(BaseConfig):

  class play:
    load_student_config = False
    mask_priv_obs = False

  class env:
    num_envs = 4096
    # num_observations = 48 + 45
    # num_observations = 235 + 10 * 180 * 320
    # num_observations = 42 + 200
    num_observations = 235 
    # num_observations = 235
    # num_observations = 48 + 32 * 32
    # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
    num_privileged_obs = None
    num_actions = 12
    env_spacing = 3.  # not used with heightfields/trimeshes 
    send_timeouts = True # send time out information to the algorithm
    episode_length_s = 20 # episode length in seconds
    obs_type = "og"

    use_camera = False
    concatenate_depth = False

  class depth: 
    height = 32
    width = 32
    dt = 0.1
    viz = False
    num_depth_frames = 10
    clip = 1
    scale = 1

  class terrain:
    mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
    horizontal_scale = 0.05 # [m]
    vertical_scale = 0.005 # [m]
    border_size = 25 # [m]
    height = [0.03, 0.05]
    # height = [0.00, 0.00]
    gap_size = [0.15, 0.20]
    stepping_stone_distance = [0.05, 0.1]
    downsampled_scale = 0.05
    curriculum = False
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.
    # rough terrain only:
    measure_heights = True
    measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                         0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    # measured_points_x = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
    # measured_points_y = [-0.4, -0.2, 0., 0.2, 0.4]
    selected = False # select a unique terrain type and pass all arguments
    terrain_kwargs = None # Dict of arguments for selected terrain
    max_init_terrain_level = 5 # starting curriculum state
    terrain_length = 8.
    terrain_width = 8.
    num_rows= 10 # number of terrain rows (levels)
    num_cols = 20 # number of terrain cols (types)
    # terrain types: [smooth slope, 
    #                 rough slope, 
    #                 rough stairs up, 
    #                 rough stairs down, 
    #                 discrete, 
    #                 stepping stones
    #                 gaps, 
    #                 smooth flat]
    terrain_proportions = [0.0, 0.15, 0.2, 0.15, 0.0, 0.0, 0.4, 0.1]
    # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # trimesh only:
    slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

  class commands:
    curriculum = False
    max_curriculum = 1.
    # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    num_commands = 4
    resampling_time = 10. # time before command are changed[s]
    heading_command = True # if true: compute ang vel command from heading error

    # Easy ranges
    class ranges:
      lin_vel_x = [0.35, 0.35] # min max [m/s]
      lin_vel_y = [0.0, 0.0]   # min max [m/s]
      ang_vel_yaw = [0, 0]    # min max [rad/s]
      heading = [0, 0]

    # Full ranges
    # class ranges:
    #     lin_vel_x = [-1.0, 1.0] # min max [m/s]
    #     lin_vel_y = [-1.0, 1.0]   # min max [m/s]
    #     ang_vel_yaw = [-1, 1]    # min max [rad/s]
    #     heading = [-3.14, 3.14]

  class init_state:
    pos = [0.0, 0.0, 1.] # x,y,z [m]
    rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
    lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
    ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
    default_joint_angles = { # target angles when action = 0.0
        "joint_a": 0., 
        "joint_b": 0.}

  class control:
    control_type = 'P' # P: position, V: velocity, T: torques
    # PD Drive parameters:
    stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
    damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation = 4

  class asset:
    file = ""
    foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
    penalize_contacts_on = []
    terminate_after_contacts_on = []
    disable_gravity = False
    # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
    collapse_fixed_joints = True
    fix_base_link = False # fixe the base of the robot
    # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
    default_dof_drive_mode = 3
    self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    # replace collision cylinders with capsules, leads to faster/more stable simulation
    replace_cylinder_with_capsule = True
    flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

    density = 0.001
    angular_damping = 0.
    linear_damping = 0.
    max_angular_velocity = 1000.
    max_linear_velocity = 1000.
    armature = 0.
    thickness = 0.01

  class domain_rand:
    randomize_friction = True
    friction_range = [0.5, 1.25]
    randomize_base_mass = False
    added_mass_range = [-1., 1.]
    push_robots = True
    push_interval_s = 15
    max_push_vel_xy = 1.

  class rewards:
    class scales:
      termination = -0.1
      tracking_lin_vel = 20.0
      tracking_ang_vel = -4.0
      lin_vel_z = -2.0
      ang_vel_xy = -0.05
      orientation = -0.
      torques = -0.0
      dof_vel = -0.
      dof_acc = -2.5e-7
      base_height = -0. 
      feet_air_time =  0.0
      collision = -1.
      feet_stumble = -0.0 
      action_rate = -0.01
      stand_still = -0.
      act_penalty = -0.1
      sideways_penalty = -0.04
      work = -0.001

    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positive_rewards = not  True
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1.
    soft_torque_limit = 1.
    base_height_target = 1.
    max_contact_force = 100. # forces above this value are penalized

  class normalization:
    class obs_scales:
      lin_vel = 2.0
      ang_vel = 0.25
      dof_pos = 1.0
      dof_vel = 0.05
      height_measurements = 5.0
    clip_observations = 100.
    clip_actions = 100.

  class noise:
    add_noise = True
    noise_level = 0.3 # scales other values
    class noise_scales:
      dof_pos = 0.01
      dof_vel = 1.5
      lin_vel = 0.1
      ang_vel = 0.2
      gravity = 0.05
      height_measurements = 0.1

  # viewer camera:
  class viewer:
    ref_env = 0
    pos = [10, 0, 6]  # [m]
    lookat = [11., 5, 3.]  # [m]

  class sim:
    dt =  0.005
    substeps = 1
    gravity = [0., 0. ,-9.81]  # [m/s^2]
    up_axis = 1  # 0 is y, 1 is z

    class physx:
      num_threads = 10
      solver_type = 1  # 0: pgs, 1: tgs
      num_position_iterations = 4
      num_velocity_iterations = 0
      contact_offset = 0.01  # [m]
      rest_offset = 0.0   # [m]
      bounce_threshold_velocity = 0.5 #0.5 [m/s]
      max_depenetration_velocity = 1.0
      max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
      default_buffer_size_multiplier = 5
      contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPOLocal(BaseConfig):
  seed = 1
  runner_class_name = 'OnPolicyRunner'

  class distil:
    num_episodes = 10000
    num_epochs = 10000
    num_teacher_obs = 235 - 12 - 24 - 3
    logging_interval = 5
    save_interval = 1000  
    epoch_save_interval = 10
    batch_size = 1024
    num_steps = 100
    num_training_iters = 10
    lr = 1e-3
    training_device = "cuda:0"
    max_buffer_length = 1000000
    num_warmup_steps = 100
  class policy:
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    use_depth_backbone = False
    num_input_vis_obs = 10 * 32 * 32
    num_output_vis_obs = 10 * 128
    # only for 'ActorCriticRecurrent':
    rnn_type = 'lstm'
    rnn_hidden_size = 512
    rnn_num_layers = 1

  class teacher_policy:
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    use_depth_backbone = False
    backbone_type = "deepgait_coordconv"
    num_input_vis_obs = 10 * 32 * 32
    num_output_vis_obs = 1280

  class student_policy:
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    use_depth_backbone = False
    backbone_type = "deepgait_coordconv"
    num_input_vis_obs = 10 * 32 * 32
    num_output_vis_obs = 1280
  class algorithm:
    # training params
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3 #5.e-4
    schedule = 'adaptive' # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.
    priv_dims = 0
    num_vis_obs = 0
    teacher_alpha = 1.0

  class teacher_runner:
    policy_class_name = 'ActorCritic'
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24 # per iteration
    max_iterations = 1500 # number of policy updates

    # logging
    save_interval = 500 # check for potential saves every this many iterations
    experiment_name = 'rough_a1'
    run_name = ''
    # load and resume
    resume = False
    load_run = -1 # -1 = last run
    checkpoint = -1 # -1 = last saved model
    resume_path = None # updated from load_run and chkpt
  class runner:
    policy_class_name = 'ActorCritic'
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24 # per iteration
    max_iterations = 1500 # number of policy updates

    # logging
    save_interval = 500 # check for potential saves every this

class A1RoughCfgLocal( LeggedRobotCfgLocal ):
  class init_state( LeggedRobotCfgLocal.init_state ):
    pos = [0.0, 0.0, 0.35] # x,y,z [m]
    default_joint_angles = { # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,   # [rad]
        'RL_hip_joint': 0.1,   # [rad]
        'FR_hip_joint': -0.1 ,  # [rad]
        'RR_hip_joint': -0.1,   # [rad]

        'FL_thigh_joint': 0.8,     # [rad]
        'RL_thigh_joint': 1.,   # [rad]
        'FR_thigh_joint': 0.8,     # [rad]
        'RR_thigh_joint': 1.,   # [rad]

        'FL_calf_joint': -1.5,   # [rad]
        'RL_calf_joint': -1.5,    # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5,    # [rad]
    }

  class control( LeggedRobotCfgLocal.control ):
    # PD Drive parameters:
    control_type = 'P'
    stiffness = {'joint': 30.}  # [N*m/rad]
    damping = {'joint': 0.5}     # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation = 4

  class asset( LeggedRobotCfgLocal.asset ):
    file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
    foot_name = "foot"
    penalize_contacts_on = ["thigh", "calf"]
    terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
    self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

  class rewards( LeggedRobotCfgLocal.rewards ):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    class scales( LeggedRobotCfgLocal.rewards.scales ):
      torques = 0.0
      dof_pos_limits = -10.0

class A1RoughCfgPPO( LeggedRobotCfgPPOLocal ):
  class algorithm( LeggedRobotCfgPPOLocal.algorithm ):
    entropy_coef = 0.01
  class runner( LeggedRobotCfgPPOLocal.runner ):
    run_name = ''
    experiment_name = 'rough_a1'

# ----- legged_gym.envs.base.legged_robot_config ----- #

class LeggedRobotCfg(BaseConfig):

  class play:
    load_student_config = False
    mask_priv_obs = False

  class env:
    num_envs = 6144

    n_scan = 132
    n_priv = 3+3 +3
    n_priv_latent = 4 + 1 + 12 +12
    n_proprio = 3 + 2 + 3 + 4 + 36 + 5
    history_len = 10

    num_observations = n_proprio + n_scan + history_len*n_proprio + \
        n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
    # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
    num_privileged_obs = None
    num_actions = 12
    env_spacing = 3.  # not used with heightfields/trimeshes 
    send_timeouts = True # send time out information to the algorithm
    episode_length_s = 20 # episode length in seconds
    obs_type = "og"
    history_encoding = True
    reorder_dofs = True

    # action_delay_range = [0, 5]
    # additional visual inputs 
    # action_delay_range = [0, 5]

    # additional visual inputs 
    include_foot_contacts = True

    randomize_start_pos = False
    randomize_start_vel = False
    randomize_start_yaw = False
    rand_yaw_range = 1.2
    randomize_start_y = False
    rand_y_range = 0.5
    randomize_start_pitch = False
    rand_pitch_range = 1.6

    contact_buf_len = 100

    next_goal_threshold = 0.2
    reach_goal_delay = 0.1
    num_future_goal_obs = 2

  class depth:
    use_camera = False
    camera_num_envs = 192
    camera_terrain_num_rows = 10
    camera_terrain_num_cols = 20

    position = [0.27, 0, 0.03]  # front camera
    angle = [-5, 5]  # positive pitch down

    update_interval = 5  # 5 works without retraining, 8 worse

    original = (106, 60)
    resized = (87, 58)
    horizontal_fov = 87
    buffer_len = 2

    near_clip = 0
    far_clip = 2
    dis_noise = 0.0

    scale = 1
    invert = True

  class normalization:
    class obs_scales:
      lin_vel = 2.0
      ang_vel = 0.25
      dof_pos = 1.0
      dof_vel = 0.05
      height_measurements = 5.0
    clip_observations = 100.
    clip_actions = 1.2

  class noise:
    add_noise = False
    noise_level = 1.0 # scales other values
    quantize_height = True
    class noise_scales:
      rotation = 0.0
      dof_pos = 0.01
      dof_vel = 0.05
      lin_vel = 0.05
      ang_vel = 0.05
      gravity = 0.02
      height_measurements = 0.02

  class terrain:
    mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
    hf2mesh_method = "grid"  # grid or fast
    max_error = 0.1 # for fast
    max_error_camera = 2

    y_range = [-0.4, 0.4]

    edge_width_thresh = 0.05
    horizontal_scale = 0.05 # [m] influence computation time by a lot
    horizontal_scale_camera = 0.1
    vertical_scale = 0.005 # [m]
    border_size = 5 # [m]
    height = [0.02, 0.06]
    simplify_grid = False
    gap_size = [0.02, 0.1]
    stepping_stone_distance = [0.02, 0.08]
    downsampled_scale = 0.075
    curriculum = True

    all_vertical = False
    no_flat = True

    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.
    measure_heights = True
    measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6,
                         0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
    measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
    measure_horizontal_noise = 0.0

    selected = False # select a unique terrain type and pass all arguments
    terrain_kwargs = None # Dict of arguments for selected terrain
    max_init_terrain_level = 5 # starting curriculum state
    terrain_length = 18.
    terrain_width = 4
    num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
    num_cols = 40 # number of terrain cols (types)

    terrain_dict = {"smooth slope": 0., 
                    "rough slope up": 0.0,
                    "rough slope down": 0.0,
                    "rough stairs up": 0., 
                    "rough stairs down": 0., 
                    "discrete": 0., 
                    "stepping stones": 0.0,
                    "gaps": 0., 
                    "smooth flat": 0,
                    "pit": 0.0,
                    "wall": 0.0,
                    "platform": 0.,
                    "large stairs up": 0.,
                    "large stairs down": 0.,
                    "parkour": 0.2,
                    "parkour_hurdle": 0.2,
                    "parkour_flat": 0.2,
                    "parkour_step": 0.2,
                    "parkour_gap": 0.2,
                    "demo": 0.0,}
    terrain_proportions = list(terrain_dict.values())

    # trimesh only:
    slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
    origin_zero_z = True

    num_goals = 8

  class commands:
    curriculum = False
    max_curriculum = 1.
    # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    num_commands = 4
    resampling_time = 6. # time before command are changed[s]
    heading_command = True # if true: compute ang vel command from heading error

    lin_vel_clip = 0.2
    ang_vel_clip = 0.4
    # Easy ranges
    class ranges:
      lin_vel_x = [0., 1.5] # min max [m/s]
      lin_vel_y = [0.0, 0.0]   # min max [m/s]
      ang_vel_yaw = [0, 0]    # min max [rad/s]
      heading = [0, 0]

    # Easy ranges
    class max_ranges:
      lin_vel_x = [0.3, 0.8] # min max [m/s]
      lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]
      ang_vel_yaw = [-0, 0]    # min max [rad/s]
      heading = [-1.6, 1.6]

    class crclm_incremnt:
      lin_vel_x = 0.1 # min max [m/s]
      lin_vel_y = 0.1  # min max [m/s]
      ang_vel_yaw = 0.1    # min max [rad/s]
      heading = 0.5

    waypoint_delta = 0.7

  class init_state:
    pos = [0.0, 0.0, 1.] # x,y,z [m]
    rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
    lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
    ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
    default_joint_angles = { # target angles when action = 0.0
        "joint_a": 0., 
        "joint_b": 0.}

  class control:
    control_type = 'P' # P: position, V: velocity, T: torques
    # PD Drive parameters:
    stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
    damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.5
    # decimation: Number of control action updates @ sim DT per policy DT
    decimation = 4

  class asset:
    file = ""
    foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
    penalize_contacts_on = []
    terminate_after_contacts_on = []
    disable_gravity = False
    # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
    collapse_fixed_joints = True
    fix_base_link = False # fixe the base of the robot
    # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
    default_dof_drive_mode = 3
    self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    # replace collision cylinders with capsules, leads to faster/more stable simulation
    replace_cylinder_with_capsule = True
    flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

    density = 0.001
    angular_damping = 0.
    linear_damping = 0.
    max_angular_velocity = 1000.
    max_linear_velocity = 1000.
    armature = 0.
    thickness = 0.01

  class domain_rand:
    randomize_friction = True
    friction_range = [0.6, 2.]
    randomize_base_mass = True
    added_mass_range = [0., 3.]
    randomize_base_com = True
    added_com_range = [-0.2, 0.2]
    push_robots = True
    push_interval_s = 8
    max_push_vel_xy = 0.5

    randomize_motor = True
    motor_strength_range = [0.8, 1.2]

    delay_update_global_steps = 24 * 8000
    action_delay = False
    action_curr_step = [1, 1]
    action_curr_step_scratch = [0, 1]
    action_delay_view = 1
    action_buf_len = 8

  class rewards:
    class scales:
      # tracking rewards
      tracking_goal_vel = 1.5
      tracking_yaw = 0.5
      # regularization rewards
      lin_vel_z = -1.0
      ang_vel_xy = -0.05
      orientation = -1.
      dof_acc = -2.5e-7
      collision = -10.
      action_rate = -0.1
      delta_torques = -1.0e-7
      torques = -0.00001
      hip_pos = -0.5
      dof_error = -0.04
      feet_stumble = -1
      feet_edge = -1

    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positive_rewards = True
    tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1
    soft_torque_limit = 0.4
    base_height_target = 1.
    max_contact_force = 40. # forces above this value are penalized

  # viewer camera:
  class viewer:
    ref_env = 0
    pos = [10, 0, 6]  # [m]
    lookat = [11., 5, 3.]  # [m]

  class sim:
    dt =  0.005
    substeps = 1
    gravity = [0., 0. ,-9.81]  # [m/s^2]
    up_axis = 1  # 0 is y, 1 is z

    class physx:
      num_threads = 10
      solver_type = 1  # 0: pgs, 1: tgs
      num_position_iterations = 4
      num_velocity_iterations = 0
      contact_offset = 0.01  # [m]
      rest_offset = 0.0   # [m]
      bounce_threshold_velocity = 0.5 #0.5 [m/s]
      max_depenetration_velocity = 1.0
      max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
      default_buffer_size_multiplier = 5
      contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):

  seed = 1
  runner_class_name = 'OnPolicyRunner'

  class policy:
    init_noise_std = 1.0
    continue_from_last_std = True
    scan_encoder_dims = [128, 64, 32]
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    priv_encoder_dims = [64, 20]
    activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    # only for 'ActorCriticRecurrent':
    rnn_type = 'lstm'
    rnn_hidden_size = 512
    rnn_num_layers = 1

    tanh_encoder_output = False

  class algorithm:
    # training params
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 2.e-4 #5.e-4
    schedule = 'adaptive' # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.
    # dagger params
    dagger_update_freq = 20
    priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
    priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]

  class depth_encoder:
    if_depth = LeggedRobotCfg.depth.use_camera
    depth_shape = LeggedRobotCfg.depth.resized
    buffer_len = LeggedRobotCfg.depth.buffer_len
    hidden_dims = 512
    learning_rate = 1.e-3
    num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24

  class estimator:
    train_with_estimated_states = True
    learning_rate = 1.e-4
    hidden_dims = [128, 64]
    priv_states_dim = LeggedRobotCfg.env.n_priv
    num_prop = LeggedRobotCfg.env.n_proprio
    num_scan = LeggedRobotCfg.env.n_scan

  class runner:
    policy_class_name = 'ActorCritic'
    algorithm_class_name = 'PPO'
    num_steps_per_env = 24 # per iteration
    max_iterations = 50000 # number of policy updates

    # logging
    save_interval = 100 # check for potential saves every this many iterations
    experiment_name = 'rough_a1'
    run_name = ''
    # load and resume
    resume = False
    load_run = -1 # -1 = last run
    checkpoint = -1 # -1 = last saved model
    resume_path = None # updated from load_run and chkpt

# ----- run as a script ----- #

if __name__ == "__main__":

  # default logdir
  log_dir = "luke/models/"
  device = "cpu"
  init_wandb = False

  # create dictionary of configs
  train_cfg_dict = LeggedRobotCfgPPO()

  # from legged_gym/utils/task_registry.py -> make_alg_runner
  runner = OnPolicyRunner(env, 
                          train_cfg_dict, 
                          log_dir, 
                          init_wandb=init_wandb,
                          device=device)