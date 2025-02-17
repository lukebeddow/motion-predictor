"""
Wrappers for DeepMind Control environments
"""

from repos.dreamerv3 import embodied 
import elements
import numpy as np
import os
import functools

from dm_control import manipulation
from dm_control import suite
from dm_control.locomotion.examples import basic_rodent_2020

class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      quadruped=2,
      rodent=4,
  )

  def __init__(
      self, env, repeat=1, size=(64, 64), proprio=True, image=True, camera=-1):
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        env = manipulation.load(task + '_vision')
      elif domain == 'rodent':
        # camera 0: topdown map
        # camera 2: shoulder
        # camera 4: topdown tracking
        # camera 5: eyes
        env = getattr(basic_rodent_2020, task)()
      else:
        env = suite.load(domain, task)
    self._dmenv = env
    self._env = FromDM(self._dmenv)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._size = size
    self._proprio = proprio
    self._image = image
    self._camera = camera

  @functools.cached_property
  def obs_space(self):
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    spaces = self._env.obs_space.copy()
    if not self._proprio:
      spaces = {k: spaces[k] for k in basic}
    key = 'image' if self._image else 'log/image'
    spaces[key] = elements.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    basic = ('is_first', 'is_last', 'is_terminal', 'reward')
    if not self._proprio:
      obs = {k: obs[k] for k in basic}
    key = 'image' if self._image else 'log/image'
    obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    for key, space in self.obs_space.items():
      if np.issubdtype(space.dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (key, obs[key])
    return obs


class FromDM(embodied.Env):

  def __init__(self, env, obs_key='observation', act_key='action'):
    self._env = env
    obs_spec = self._env.observation_spec()
    act_spec = self._env.action_spec()
    self._obs_dict = isinstance(obs_spec, dict)
    self._act_dict = isinstance(act_spec, dict)
    self._obs_key = not self._obs_dict and obs_key
    self._act_key = not self._act_dict and act_key
    self._obs_empty = []
    self._done = True

  @functools.cached_property
  def obs_space(self):
    spec = self._env.observation_spec()
    spec = spec if self._obs_dict else {self._obs_key: spec}
    if 'reward' in spec:
      spec['obs_reward'] = spec.pop('reward')
    for key, value in spec.copy().items():
      if int(np.prod(value.shape)) == 0:
        self._obs_empty.append(key)
        del spec[key]
    spaces = {
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    for key, value in spec.items():
      key = key.replace('/', '_')
      spaces[key] = self._convert(value)
    return spaces

  @functools.cached_property
  def act_space(self):
    spec = self._env.action_spec()
    spec = spec if self._act_dict else {self._act_key: spec}
    return {
        'reset': elements.Space(bool),
        **{k or self._act_key: self._convert(v) for k, v in spec.items()},
    }

  def step(self, action):
    action = action.copy()
    reset = action.pop('reset')
    if reset or self._done:
      time_step = self._env.reset()
    else:
      action = action if self._act_dict else action[self._act_key]
      time_step = self._env.step(action)
    self._done = time_step.last()
    return self._obs(time_step)

  def _obs(self, time_step):
    if not time_step.first():
      assert time_step.discount in (0, 1), time_step.discount
    obs = time_step.observation
    obs = dict(obs) if self._obs_dict else {self._obs_key: obs}
    if 'reward' in obs:
      obs['obs_reward'] = obs.pop('reward')
    for key in self._obs_empty:
      del obs[key]
    obs = {k.replace('/', '_'): v for k, v in obs.items()}
    return dict(
        reward=np.float32(0.0 if time_step.first() else time_step.reward),
        is_first=time_step.first(),
        is_last=time_step.last(),
        is_terminal=False if time_step.first() else time_step.discount == 0,
        **obs,
    )

  def _convert(self, space):
    if hasattr(space, 'num_values'):
      return elements.Space(space.dtype, (), 0, space.num_values)
    elif hasattr(space, 'minimum'):
      assert np.isfinite(space.minimum).all(), space.minimum
      assert np.isfinite(space.maximum).all(), space.maximum
      return elements.Space(
          space.dtype, space.shape, space.minimum, space.maximum)
    else:
      return elements.Space(space.dtype, space.shape, None, None)

def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)
  env = embodied.wrappers.UnifyDtypes(env)
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)
  return env
