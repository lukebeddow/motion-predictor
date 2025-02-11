import numpy as np
from dataclasses import dataclass, asdict
from collections import namedtuple, deque
from copy import deepcopy
import random
import itertools
from math import floor, ceil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torchaudio.functional import lfilter

# ----- helper functions, networks, data buffers ----- #

def combined_shape(length, shape=None):
  if shape is None:
    return (length,)
  return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount, device):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    b_coeff = torch.tensor([1, 0], device=device)
    a_coeff = torch.tensor([1, float(-discount)], device=device)
    waveform = x.flip(dims=(-1,))
    return lfilter(waveform, a_coeff, b_coeff).flip(dims=(-1,))

def mlp(sizes, activation, output_activation=nn.Identity):
  layers = []
  for j in range(len(sizes)-1):
    act = activation if j < len(sizes)-2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
  return nn.Sequential(*layers)

class PPOBuffer:

  def __init__(self, capacity, obs_dim, action_dim, device, gamma, lam):
    self.device = device
    self.capacity = capacity
    self.states = torch.zeros(combined_shape(capacity, obs_dim), dtype=torch.float32, device=device)
    self.actions = torch.zeros(combined_shape(capacity, action_dim), dtype=torch.float32, device=device)
    self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.logprobs = torch.zeros(capacity, dtype=torch.float32, device=device)
    self.gamma = gamma
    self.lam = lam
    self.index = 0
    self.trajectory_start_idx = 0
    self.capacity = capacity

  def push(self, state, action, reward, value, logprob):
    """
    Save a transition, assume that all incoming argument are tensor([x])
    """

    assert self.index < self.capacity
    self.states[self.index] = state[0]
    self.actions[self.index] = action[0]
    self.rewards[self.index] = reward[0]
    self.values[self.index] = value[0]
    self.logprobs[self.index] = logprob[0]
    self.index += 1

  def to_torch(self, data, dtype=None):
    """
    Convert some data to a torch tensor([x])
    """
    if dtype == None: dtype = torch.float32
    return torch.tensor(data, device=self.device, dtype=dtype).unsqueeze(0)

  def __len__(self):
    return len(self.capacity)

  def all_to(self, device):
    """
    Move entire buffer to a device
    """
    self.device = device
    self.states = self.states.to(torch.device(device))
    self.actions = self.actions.to(torch.device(device))
    self.advantages = self.advantages.to(torch.device(device))
    self.rewards = self.rewards.to(torch.device(device))
    self.returns = self.returns.to(torch.device(device))
    self.values = self.values.to(torch.device(device))
    self.logprobs = self.logprobs.to(torch.device(device))

  def finish_trajectory(self, last_val=0):
    """
    Call this at the end of a trajectory, or when one gets cut off
    by an epoch ending. This looks back in the buffer to where the
    trajectory started, and uses rewards and value estimates from
    the whole trajectory to compute advantage estimates with GAE-Lambda,
    as well as compute the rewards-to-go for each state, to use as
    the targets for the value function.

    The "last_val" argument should be 0 if the trajectory ended
    because the agent reached a terminal state (died), and otherwise
    should be V(s_T), the value function estimated for the last state.
    This allows us to bootstrap the reward-to-go calculation to account
    for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
    """

    trajectory_slice = slice(self.trajectory_start_idx, self.index)
    rewards = torch.cat((self.rewards[trajectory_slice], torch.tensor([last_val], device=self.device)))
    values = torch.cat((self.values[trajectory_slice], torch.tensor([last_val], device=self.device)))

    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
    self.advantages[trajectory_slice] = discount_cumsum(deltas, self.gamma * self.lam, self.device)

    # the next line computes rewards-to-go, to be targets for the value function
    self.returns[trajectory_slice] = discount_cumsum(rewards, self.gamma, self.device)[:-1]
    
    self.path_start_idx = self.index

  def get(self):
    """
    Call this at the end of an epoch to get all of the data from
    the buffer, with advantages appropriately normalized (shifted to have
    mean zero and std one). Also, resets some pointers in the buffer.
    """
    assert self.index == self.capacity    # buffer has to be full before you can get
    
    # reset before the next trajectory
    self.index = 0
    self.path_start_idx = 0

    # print("Buffer devices")
    # print("obs.device", self.states[0].device, self.states[-1].device)
    # print("act.device", self.actions[0].device, self.actions[-1].device)
    # print("ret.device", self.returns[0].device, self.returns[-1].device)
    # print("adv.device", self.advantages[0].device, self.advantages[-1].device)

    # the next two lines implement the advantage normalization trick
    adv_std, adv_mean = torch.std_mean(self.advantages)
    self.advantages = (self.advantages - adv_mean) / adv_std
    data = dict(obs=self.states, act=self.actions, ret=self.returns,
                adv=self.advantages, logp=self.logprobs)
    return data

# ----- base actor critic network building blocks ----- #

class Actor(nn.Module):

  def _distribution(self, obs):
    raise NotImplementedError

  def _log_prob_from_distribution(self, pi, act):
    raise NotImplementedError

  def forward(self, obs, act=None):
    # Produce action distributions for given observations, and 
    # optionally compute the log likelihood of given actions under
    # those distributions.
    pi = self._distribution(obs)
    logp_a = None
    if act is not None:
      logp_a = self._log_prob_from_distribution(pi, act)
    return pi, logp_a

class MLPCategoricalActor(Actor):
    
  def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device="cpu"):
    super().__init__()
    self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.to_device(device)

  def _distribution(self, obs):
    logits = self.logits_net(obs)
    return Categorical(logits=logits)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.logits_net.to(device)
    self.device = device

class MLPGaussianActor(Actor):

  def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device="cpu"):
    super().__init__()
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.to_device(device)

  def _distribution(self, obs):
    mu = self.mu_net(obs)
    # log_std stays on cpu so we need to move this to our device
    std = torch.exp(self.log_std).to(self.device)
    return Normal(mu, std)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.mu_net.to(device)
    self.device = device

class MLPCritic(nn.Module):

  def __init__(self, obs_dim, hidden_sizes, activation, device="cpu"):
    super().__init__()
    self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    self.to_device(device)

  def forward(self, obs):
    return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.v_net.to(device)
    self.device = device

class MLPActorCriticPG(nn.Module):

  name = "MLPActorCriticPG_"

  def __init__(self, n_obs, action_dim, continous_actions=True,
                hidden_sizes=(64,64), activation=nn.Tanh, mode="train",
                device="cpu"):
    super().__init__()

    self.n_obs = n_obs
    self.n_actions = action_dim
    self.mode = mode
    self.device = device

    # policy builder depends on action space
    if continous_actions:
      self.action_dim = action_dim
      self.pi = MLPGaussianActor(n_obs, action_dim, hidden_sizes, activation, device=device)
    else:
      self.action_dim = 1 # discrete so only one action
      self.pi = MLPCategoricalActor(n_obs, action_dim, hidden_sizes, activation, device=device)

    # build value function
    self.vf  = MLPCritic(n_obs, hidden_sizes, activation)

    if self.mode not in ["test", "train"]:
      raise RuntimeError(f"MLPActorCriticPG given mode={mode}, should be 'test' or 'train'")

    # add hidden layer size into the name
    for i in range(len(hidden_sizes)):
      if i == 0: self.name += f"{hidden_sizes[i]}"
      if i > 0: self.name += f"x{hidden_sizes[i]}"

  def step(self, obs):
    with torch.no_grad():
      pi = self.pi._distribution(obs)
      a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      val = self.vf(obs)
    return a, val, logp_a

  def act(self, obs):
    return self.step(obs)[0]

  def set_device(self, device):
    self.pi.to_device(device)
    self.vf.to_device(device)
    self.device = device

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.pi.train()
    self.vf.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.pi.eval()
    self.vf.eval()

# ----- CNN and multi-input actor critics ----- #

class CNNCategoricalActor(Actor):
    
  def __init__(self, img_dim, sensor_dim, act_dim, device):
    super().__init__()
    self.logits_net = MixedNetwork(sensor_dim, img_dim, act_dim)
    self.name = self.logits_net.name
    self.device = device
    self.logits_net.to_device(device)

  def _distribution(self, obs):
    logits = self.logits_net(obs)
    return Categorical(logits=logits)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)
  
  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.logits_net.to(device)
    self.device = device

class CNNGaussianActor(Actor):

  def __init__(self, img_dim, sensor_dim, act_dim, device):
    super().__init__()
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    self.mu_net = MixedNetwork(sensor_dim, img_dim, act_dim)
    self.name = self.mu_net.name
    self.device = device
    self.mu_net.to_device(device)

  def _distribution(self, obs):
    mu = self.mu_net(obs)
    # log_std stays on cpu so we need to move this to our device
    std = torch.exp(self.log_std).to(self.device)
    return Normal(mu, std)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.mu_net.to_device(device)
    self.mu_net.to(device)
    self.device = device

class CNNCritic(nn.Module):

  def __init__(self, img_dim, sensor_dim, device):
    super().__init__()
    self.v_net = MixedNetwork(sensor_dim, img_dim, 1) # act dim = 1
    self.device = device
    self.v_net.to_device(device)

  def forward(self, obs):
    return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.v_net.to_device(device)
    self.v_net.to(device)
    self.device = device

class CNNActorCriticPG(nn.Module):

  name = "CNNActorCriticPG_"

  def __init__(self, img_size, sensor_dim, action_dim, continous_actions=True,
               mode="train", device="cuda"):
    super().__init__()

    self.img_size = img_size
    self.sensor_dim = sensor_dim
    self.n_actions = action_dim
    self.mode = mode
    self.device = device

    self.n_obs = [img_size[0] + 1, img_size[1], img_size[2]]

    # policy builder depends on action space
    if continous_actions:
      self.action_dim = action_dim
      self.pi = CNNGaussianActor(img_size, sensor_dim, action_dim, device=device)
    else:
      self.action_dim = 1 # discrete so only one action
      self.pi = CNNCategoricalActor(img_size, sensor_dim, action_dim, device=device)

    # build value function
    self.vf  = CNNCritic(img_size, sensor_dim, device=device)

    if self.mode not in ["test", "train"]:
      raise RuntimeError(f"CNNActorCriticPG given mode={mode}, should be 'test' or 'train'")

    # # add hidden layer size into the name
    # for i in range(len(hidden_sizes)):
    #   if i == 0: self.name += f"{hidden_sizes[i]}"
    #   if i > 0: self.name += f"x{hidden_sizes[i]}"

    self.name += self.pi.name

  def step(self, obs):
    with torch.no_grad():
      pi = self.pi._distribution(obs)
      a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      val = self.vf(obs)
    return a, val, logp_a

  def act(self, obs):
    return self.step(obs)[0]

  def set_device(self, device):
    self.pi.to_device(device)
    self.vf.to_device(device)
    self.device = device

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.pi.train()
    self.vf.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.pi.eval()
    self.vf.eval()

class NetCategoricalActor(Actor):
    
  def __init__(self, network, img_dim, sensor_dim, act_dim, device, netargs={}):
    super().__init__()
    self.logits_net = network(sensor_dim, img_dim, act_dim, **netargs)
    self.name = self.logits_net.name
    self.device = device
    self.logits_net.to_device(device)

  def _distribution(self, obs):
    logits = self.logits_net(obs)
    return Categorical(logits=logits)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act)
  
  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.logits_net.to(device)
    self.device = device

class NetGaussianActor(Actor):

  def __init__(self, network, img_dim, sensor_dim, act_dim, device, netargs={}):
    super().__init__()
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    self.mu_net = network(sensor_dim, img_dim, act_dim, **netargs)
    self.name = self.mu_net.name
    self.device = device
    self.mu_net.to_device(device)

  def _distribution(self, obs):
    mu = self.mu_net(obs)
    # log_std stays on cpu so we need to move this to our device
    std = torch.exp(self.log_std).to(self.device)
    return Normal(mu, std)

  def _log_prob_from_distribution(self, pi, act):
    return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.mu_net.to(device)
    self.device = device

class NetCritic(nn.Module):

  def __init__(self, network, img_dim, sensor_dim, device, netargs={}):
    super().__init__()
    self.v_net = network(sensor_dim, img_dim, 1, **netargs) # act dim = 1
    self.device = device
    self.v_net.to_device(device)

  def forward(self, obs):
    return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

  def to_device(self, device=None):
    if device is None:
      device = self.device
    self.v_net.to(device)
    self.device = device

class NetActorCriticPG(nn.Module):

  name = "NetActorCriticPG"

  def __init__(self, network, img_size, sensor_dim, action_dim, continous_actions=True,
               mode="train", device="cuda", netargs={}):
    super().__init__()

    self.img_size = img_size
    self.sensor_dim = sensor_dim
    self.n_actions = action_dim
    self.mode = mode
    self.device = device

    self.n_obs = [img_size[0] + 1, img_size[1], img_size[2]]

    # policy builder depends on action space
    if continous_actions:
      self.action_dim = action_dim
      self.pi = NetGaussianActor(network, img_size, sensor_dim, action_dim, device=device,
                                 netargs=netargs)
    else:
      self.action_dim = 1 # discrete so only one action
      self.pi = NetCategoricalActor(network, img_size, sensor_dim, action_dim, device=device,
                                    netargs=netargs)

    # build value function
    self.vf  = NetCritic(network, img_size, sensor_dim, device=device, netargs=netargs)

    if self.mode not in ["test", "train"]:
      raise RuntimeError(f"NetActorCriticPG given mode={mode}, should be 'test' or 'train'")

    # # add hidden layer size into the name
    # for i in range(len(hidden_sizes)):
    #   if i == 0: self.name += f"{hidden_sizes[i]}"
    #   if i > 0: self.name += f"x{hidden_sizes[i]}"

    self.name += self.pi.name

  def step(self, obs):
    with torch.no_grad():
      pi = self.pi._distribution(obs)
      a = pi.sample()
      logp_a = self.pi._log_prob_from_distribution(pi, a)
      val = self.vf(obs)
    return a, val, logp_a

  def act(self, obs):
    return self.step(obs)[0]

  def set_device(self, device):
    self.pi.to_device(device)
    self.vf.to_device(device)
    self.device = device

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.pi.train()
    self.vf.train()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.pi.eval()
    self.vf.eval()

# ----- proper agents ----- #

class Agent_PPO:

  name = "Agent_PPO"

  @dataclass
  class Parameters:

    # key learning hyperparameters
    learning_rate_pi: float = 1e-4
    learning_rate_vf: float = 1e-4
    gamma: float = 0.99
    steps_per_epoch: int = 4000
    clip_ratio: float = 0.2
    train_pi_iters: int = 80
    train_vf_iters: int = 80
    lam: float = 0.97
    target_kl: float = 0.01
    max_kl_ratio: float = 1.5
    use_random_action_noise: bool = True
    random_action_noise_size: float = 0.2
    optimiser: str = "adam" # adam/adamW/RMSProp
    use_kl_penalty: bool = False
    use_entropy_regularisation: bool = False
    kl_penalty_coefficient: float = 0.2
    entropy_coefficient: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    grad_clamp_value: float = None
    # loss_criterion: str = "MSELoss" # smoothL1Loss/MSELoss/Huber

    def update(self, newdict, hard=True):
      for key, value in newdict.items():
        if hasattr(self, key):
          setattr(self, key, value)
        elif hard: raise RuntimeError(f"incorrect key: {key}")
      
  Transition = namedtuple('Transition',
                          ('state', 'action', 'reward', 'advantage', 'log_prob', 'terminal'))

  def __init__(self, 
               learning_rate_pi: float,
               learning_rate_vf: float,
               gamma: float,
               steps_per_epoch: int,
               clip_ratio: float,
               train_pi_iters: int,
               train_vf_iters: int,
               lam: float,
               target_kl: float,
               max_kl_ratio: float,
               use_random_action_noise: bool,
               random_action_noise_size: float,
               optimiser: str,
               use_kl_penalty: bool,
               use_entropy_regularisation: bool,
               kl_penalty_coefficient: float,
               entropy_coefficient: float,
               adam_beta1: float,
               adam_beta2: float,
               grad_clamp_value: float,
               device: str = "cpu", 
               rngseed: int = None, 
               mode: str = "train", 
               steps: int = None, 
               debug: bool =False):
    
    """
    Soft actor-critic agent for a given environment
    
    """

    self.params = Agent_PPO.Parameters() # legacy setup 

    self.params.learning_rate_pi = learning_rate_pi
    self.params.learning_rate_vf = learning_rate_vf
    self.params.gamma = gamma
    self.params.steps_per_epoch = steps_per_epoch
    self.params.clip_ratio = clip_ratio
    self.params.train_pi_iters = train_pi_iters
    self.params.train_vf_iters = train_vf_iters
    self.params.lam = lam
    self.params.target_kl = target_kl
    self.params.max_kl_ratio = max_kl_ratio
    self.params.use_random_action_noise = use_random_action_noise
    self.params.random_action_noise_size = random_action_noise_size
    self.params.optimiser = optimiser
    self.params.use_kl_penalty = use_kl_penalty
    self.params.use_entropy_regularisation = use_entropy_regularisation
    self.params.kl_penalty_coefficient = kl_penalty_coefficient
    self.params.entropy_coefficient = entropy_coefficient
    self.params.adam_beta1 = adam_beta1
    self.params.adam_beta2 = adam_beta2
    self.params.grad_clamp_value = grad_clamp_value

    self.device = torch.device(device)
    self.rngseed = rngseed
    self.rng = np.random.default_rng(rngseed)
    self.mode = mode
    self.steps_done = 0
    self.debug = debug
    if steps is not None:
      self.params.steps_per_epoch = steps

  def init(self, network, obs_dim=None, action_dim=None, device=None):
    """
    Main initialisation of the agent, applies settings and creates network. This
    function can be called with network='loaded' to initialise settings but not
    reinitialise the networks
    """

    if network != "loaded":

      if obs_dim is None: obs_dim = network.n_obs
      if action_dim is None: action_dim = network.action_dim

      self.mlp_ac = network
      self.action_dim = action_dim
      self.n_actions = network.n_actions
      self.n_obs = obs_dim

    # create a fresh buffer for storing trajectories, rewards, actions etc
    self.buffer = PPOBuffer(self.params.steps_per_epoch, self.n_obs, self.action_dim, 
                            self.device, self.params.gamma, self.params.lam)

    # set the network to the given device
    if device is not None: self.device = device
    self.set_device(self.device)

    if self.params.optimiser.lower() == "rmsprop":
      self.vf_optimiser = torch.optim.RMSprop(self.mlp_ac.vf.parameters(), 
                                             lr=self.params.learning_rate_vf)
      self.pi_optimiser = torch.optim.RMSprop(self.mlp_ac.pi.parameters(),
                                              lr=self.params.learning_rate_pi)
    elif self.params.optimiser.lower() == "adam":
      self.vf_optimiser = torch.optim.Adam(self.mlp_ac.vf.parameters(),
                                          lr=self.params.learning_rate_vf,
                                          betas=(self.params.adam_beta1,
                                                 self.params.adam_beta2))
      self.pi_optimiser = torch.optim.Adam(self.mlp_ac.pi.parameters(),
                                           lr=self.params.learning_rate_pi,
                                           betas=(self.params.adam_beta1,
                                                  self.params.adam_beta2))
    elif self.params.optimiser.lower() == "adamw":
      self.vf_optimiser = torch.optim.AdamW(self.mlp_ac.vf.parameters(),
                                           lr=self.params.learning_rate_vf,
                                           betas=(self.params.adam_beta1,
                                                  self.params.adam_beta2),
                                           amsgrad=True)
      self.pi_optimiser = torch.optim.AdamW(self.mlp_ac.pi.parameters(),
                                            lr=self.params.learning_rate_pi,
                                            betas=(self.params.adam_beta1,
                                                   self.params.adam_beta2),
                                            amsgrad=True)
      
    else: raise RuntimeError(f"Invalid optimiser choice '{self.params.optimiser}', options are 'adam' or 'rmsprop'")

    # if self.params.loss_criterion.lower() == "smoothl1loss":
    #   self.loss_criterion = nn.SmoothL1Loss()
    # elif self.params.loss_criterion.lower() == "mseloss":
    #   self.loss_criterion = nn.MSELoss()
    # elif self.params.loss_criterion.lower() == "huber":
    #   self.loss_criterion = nn.HuberLoss()

    self.seed()

    if self.mode == "train": self.training_mode()
    elif self.mode == "test": self.testing_mode()
    else: raise RuntimeError(f"Agent_DQN given mode={self.mode} but valid options are 'test' and 'train'")
  
  def set_device(self, device):
    """
    Set whether on cpu or gpu, and move all the memory replay buffer to that device
    """

    if device == "cuda" and not torch.cuda.is_available(): 
      print("Agent_PPO.set_device() received request for 'cuda', but torch.cuda.is_available() == False, hence using cpu")
      device = "cpu"

    self.device = torch.device(device)

    self.buffer.all_to(self.device)
    self.mlp_ac.set_device(self.device)

    # if hasattr(self, "pi_optimiser"):
    #   # extra safety: https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
    #   for state in self.vf_optimiser.state.values():
    #     for k, v in state.items():
    #       if isinstance(v, torch.Tensor):
    #         state[k] = v.to(self.device)
    #   for state in self.pi_optimiser.state.values():
    #     for k, v in state.items():
    #       if isinstance(v, torch.Tensor):
    #         state[k] = v.to(self.device)

  def seed(self, rngseed=None):
    """
    Set a random seed for the entire environment
    """
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: rngseed = np.random.randint(0, 2_147_483_647)
    self.rngseed = rngseed
    self.rng = np.random.default_rng(rngseed)

  def select_action(self, state, decay_num, test=False):
    """
    Choose action values, a vector of values which should be [-1, +1]

    This function should return a tensor([x, y, z, ...])
    """

    # # determine if we will act randomly
    # random_action = False
    # if not test:
    #   rand = self.rng.random()
    #   eps_threshold = (self.params.eps_end 
    #     + (self.params.eps_start - self.params.eps_end)
    #     * (np.exp(-1. * decay_num / float(self.params.eps_decay))))
    #   if rand < eps_threshold: random_action = True

    # if decay_num < self.params.random_start_episodes:
    #   return torch.tensor([2*self.rng.random() - 1 for x in range(self.action_dim)], dtype=torch.float32)

    if self.mode == "test": test = True

    action, value, logprob = self.mlp_ac.step(state)

    if not test and self.params.use_random_action_noise:
      a = self.params.random_action_noise_size
      noise = 2 * a * self.rng.random((1, len(action[0]))) - a
      # print("action is", action)
      # print("noise is", noise)
      action += torch.tensor(noise, device=action.device)
      # print("result is", action)

    # print("action", action.shape)
    # print("logprob", logprob.shape)
    # print("value", value.shape)

    # store values from before the environment steps to put into buffer
    self.last_state = state
    self.last_value = value
    self.last_logprob = logprob

    return action.squeeze(0) # from tensor([[x]]) to tensor([x])
      
  def get_save_state(self):
    """
    Return the state of the model that should be saved
    """

    to_save = {
      "name" : self.name,
      "parameters" : self.params,
      "buffer" : self.buffer,
      "network" : self.mlp_ac, # do I want to save only the state dict?
      "optimiser_state_dict" : {
         "vf_optimiser" : self.vf_optimiser.state_dict(),
         "pi_optimiser" : self.pi_optimiser.state_dict()
      }
    }

    return to_save
  
  def load_save_state(self, saved_dict, device="cpu"):
    """
    Load the agent with a given saved state
    """

    # check we are compatible
    if self.name != saved_dict["name"]:
      raise RuntimeError(f"{self.name}.load_save_state() failed as given save state has agent name: {saved_dict['name']}")
    
    # input the saved data
    self.params = saved_dict["parameters"]
    self.init(saved_dict["network"], device=device)
    self.buffer = saved_dict["buffer"]
    self.vf_optimiser.load_state_dict(saved_dict["optimiser_state_dict"]["vf_optimiser"])
    self.pi_optimiser.load_state_dict(saved_dict["optimiser_state_dict"]["pi_optimiser"])

    # move to the correct device
    self.set_device(device)

    # prepare class variables
    self.steps_done = self.buffer.index

  def get_params_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
      "network_name" : self.mlp_ac.name
    })
    return param_dict

  def compute_loss_v(self, data):
    """
    Compute the value function loss
    """
    obs, ret = data['obs'], data['ret']
    return ((self.mlp_ac.vf(obs) - ret)**2).mean()
  
  def compute_loss_pi(self, data):
    """
    Compute PPO policy loss
    """
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # Policy loss
    pi, logp = self.mlp_ac.pi(obs, act)
    ratio = torch.exp(logp - logp_old)

    # are we using PPO penalty
    if self.params.use_kl_penalty:
      kl_divergence = (logp_old - logp).mean()
      loss_pi = -(ratio * adv).mean() + self.params.kl_penalty_coefficient * kl_divergence

    # else we are using PPO clip
    else:
      clip_adv = torch.clamp(ratio, 1 - self.params.clip_ratio, 1 + self.params.clip_ratio) * adv
      loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # are we using entropy regularization
    if self.params.use_entropy_regularisation:
      entropy = pi.entropy().mean()
      loss_pi -= self.params.entropy_coefficient * entropy

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    clipped = ratio.gt(1 + self.params.clip_ratio) | ratio.lt(1 - self.params.clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

    return loss_pi, pi_info

  def optimise_model(self):

    if self.debug:
      t1 = time.perf_counter()
      print("Agent.optimise_model() now running ...", end="", flush=True)

    data = self.buffer.get()

    pi_l_old, pi_info_old = self.compute_loss_pi(data)
    pi_l_old = pi_l_old.item()
    v_l_old = self.compute_loss_v(data).item()

    # Train policy with multiple steps of gradient descent
    for i in range(self.params.train_pi_iters):
      self.pi_optimiser.zero_grad()
      loss_pi, pi_info = self.compute_loss_pi(data)
      kl = pi_info['kl']
      if kl > self.params.max_kl_ratio * self.params.target_kl:
        # print('Early stopping at step %d due to reaching max kl.'%i)
        break
      loss_pi.backward()
      if self.params.grad_clamp_value is not None:
        torch.nn.utils.clip_grad_value_(self.mlp_ac.pi.parameters(), self.params.grad_clamp_value)
      self.pi_optimiser.step()

    # logger.store(StopIter=i)

    # Value function learning
    for i in range(self.params.train_vf_iters):
      self.vf_optimiser.zero_grad()
      loss_v = self.compute_loss_v(data)
      loss_v.backward()
      if self.params.grad_clamp_value is not None:
        torch.nn.utils.clip_grad_value_(self.mlp_ac.vf.parameters(), self.params.grad_clamp_value)
      self.vf_optimiser.step()

    # Log changes from update
    # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    # logger.store(LossPi=pi_l_old, LossV=v_l_old,
    #               KL=kl, Entropy=ent, ClipFrac=cf,
    #               DeltaLossPi=(loss_pi.item() - pi_l_old),
    #               DeltaLossV=(loss_v.item() - v_l_old))
      
    if self.debug:
      t2 = time.perf_counter()
      print(f"finished after {(t2 - t1):.3f} seconds")

  def update_step(self, state, action, next_state, reward, terminal, truncated):
    """
    Run this every training action step to update the model
    """

    if self.mode != "train": return

    # print("state", state.shape)
    # print("action", action.shape)
    # print("next_state", next_state.shape)
    # print("reward", reward.shape)
    # print("terminal", terminal.shape)

    # save buffer, but we use the state/value/logprob from before the action was
    # applied in the environment, we will get this 'state' next step
    self.buffer.push(
      self.last_state,
      action,
      reward,
      self.last_value,
      self.last_logprob
    )

    self.steps_done += 1

    epoch_ended = (self.steps_done >= self.params.steps_per_epoch)
    terminal = bool(terminal.item())

    # if trajectory did not reach terminal state, bootstrap value target
    if truncated or epoch_ended:
      _, end_value, _ = self.mlp_ac.step(self.last_state)
    else:
      end_value = 0

    # add the trajectory to the buffer
    self.buffer.finish_trajectory(end_value)

    # update the model and reset the epoch step counter
    if epoch_ended:
      self.optimise_model()
      self.steps_done = 0

  def update_episode(self, i_episode, finished=False):
    """
    Run this at the end of every training episode
    """
    pass

  def training_mode(self):
    """
    Put the agent into training mode
    """
    self.mode = "train"
    self.mlp_ac.training_mode()

  def testing_mode(self):
    """
    Put the agent in testing mode
    """
    self.mode = "test"
    self.mlp_ac.testing_mode()

if __name__ == "__main__":

  # test this script here
  pass
