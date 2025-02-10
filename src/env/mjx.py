import distutils.util
import os
import subprocess
from mujoco_playground import registry
import jax
import random

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
# from: https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb#scrollTo=ObF1UXrkb0Nd
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

class MJXEnv():

  def __init__(self, env_name, rngseed=None):
    """
    Wrapper for gym environments to make them compatible with our code
    """
    self.name = env_name
    self.env = registry.load(self.name)
    self.env_config = registry.get_default_config(self.name)

    # jit the key environment functions
    self.jit_reset = jax.jit(self.env.reset)
    self.jit_step = jax.jit(self.env.step)

    # set the random properties for the environment
    self.seed(rngseed)

    # extract key information
    self.act_dim = self.env.action_size
    self.obs_dim = self.env.observation_size
    
    # these currently mean nothing, check this
    self.continuous_actions = False
    self.device = "cpu"

    print(self.get_params_dict())

  def step(self, action):
    """
    Input action should be a jax.numpy.array

    state is a dataclass:
      - data: mjx.Data (mujoco data structure, not including mjModel)
      - obs: Observation (Union[jax.Array, Mapping[str, jax.Array]])
      - reward. jax.Array (float32)
      - done: jax.Array (float32)
      - metrics: Dict[str, jax.Array]
      - info: Dict[str, Any]
    """
    self.state = self.jit_step(self.state, action)
    return (
      self.state.obs,         # observation
      self.state.reward,      # reward
      self.state.done,        # terminal
      jax.numpy.array([0]),   # truncated
      self.state.info,        # info dict
    )
  
  def reset(self, rngseed=None):
    self.key, local_key = jax.random.split(self.key)
    self.state = self.jit_reset(rng=local_key)
    return self.state.obs
  
  def seed(self, rngseed=None):

    if not hasattr(self, "rngseed"): self.rngseed = rngseed
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: rngseed = random.randint(0, 2_147_483_647)

    # save the initial seed, then create the jax prng key
    self.rngseed = rngseed
    self.key = jax.random.key(self.rngseed)
    self.reset()

  def close(self):
    return

  def render(self):
    # todo
    self.env.render()

  def get_params_dict(self):
    return {
      "act_dim" : self.act_dim,
      "obs_dim" : self.obs_dim,
      "env_type" : "mujoco-playground",
      "env_config" : self.env_config,
    }
  
  def get_save_state(self):
    return {
      "name" : self.name,
      "act_dim" : self.act_dim,
      "obs_dim" : self.obs_dim,
      "rngseed" : self.rngseed,
      "env_config" : self.env_config,
    }
  
  def load_save_state(self, state_dict):

    # load the class variables from the given dictionary
    self.name = state_dict["name"]
    self.act_dim = state_dict["act_dim"]
    self.obs_dim = state_dict["obs_dim"]
    self.rngseed = state_dict["rngseed"]
    self.env_config = state_dict["env_config"]
    
    # now re-make the environment
    self.env = registry.load(self.name, self.env_config)

    # re-seed
    self.seed()

if __name__ == "__main__":

  env = MJXEnv("CartpoleBalance")

  for i in range(2):
    # action = random.random()
    # action = jax.numpy.array(action)
    # env.step(action)
    # print(env.state)

    action = []
    for j in range(env.act_dim):
      action.append(
          jax.numpy.sin(
              env.state.data.time * 2 * jax.numpy.pi * 0.5 + j * 2 * jax.numpy.pi / env.env.action_size
          )
      )
    action = jax.numpy.array(action)
    env.step(action)
    print(env.state)