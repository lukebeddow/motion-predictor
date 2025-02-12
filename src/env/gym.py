import gymnasium as gym
import random

# wrapper to use gym environments
class GymHandler():

  def __init__(self, name, seed=None):
    """
    Wrapper for gym environments to make them compatible with our code
    """
    self.name = name
    self.env = gym.make(name)
    obs, info = self.env.reset()
    self.act_dim = self.env.action_space.shape[0]
    self.obs_dim = len(obs)
    self.rngseed = seed
    self.seed()
    self.continuous_actions = False
    self.device = "cpu"

  def step(self, action):
    return self.env.step(action.to(self.device).numpy())
  
  def reset(self, rngseed=None):
    obs, info = self.env.reset(seed=rngseed)
    return obs
  
  def seed(self, rngseed=None):
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: rngseed = random.randint(0, 2_147_483_647)
    self.rngseed = rngseed
    self.env.action_space.seed(self.rngseed)
    self.env.reset(seed=self.rngseed)

  def close(self):
    self.env.close()

  def render(self):
    self.env.render()

  def get_params_dict(self):
    return {
      "act_dim" : self.act_dim,
      "obs_dim" : self.obs_dim,
      "env_type" : "gym env"
    }
  
  def get_save_state(self):
    return {
      "name" : self.name,
      "act_dim" : self.act_dim,
      "obs_dim" : self.obs_dim,
      "device" : self.device,
      "rngseed" : self.rngseed,
      "continuous_actions" : self.continuous_actions,
    }
  
  def load_save_state(self, state_dict):

    # load the class variables from the given dictionary
    self.name = state_dict["name"]
    self.act_dim = state_dict["act_dim"]
    self.obs_dim = state_dict["obs_dim"]
    self.device = state_dict["device"]
    self.rngseed = state_dict["rngseed"]
    self.continuous_actions = state_dict["continuous_actions"]
    
    # now re-make the environment
    self.env = gym.make(self.name)
