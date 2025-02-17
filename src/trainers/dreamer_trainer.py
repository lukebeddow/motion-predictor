"""
Training Dreamer, adapted from https://github.com/danijar/dreamerv3/tree/main

"""

import importlib
import os
import pathlib
import sys
from functools import partial


# folder = pathlib.Path(__file__).parent
# sys.path.insert(0, str())
# sys.path.insert(1, str(folder.parent.parent))
# __package__ = folder.name

import elements
import embodied 

import numpy as np
import portal
import ruamel.yaml as yaml
import collections
import numpy as np

class DreamerTrainer():

  def __init__(self, config:dict, env):
    self.agent = make_agent(config, act_space=env.act_space, obs_space=env.obs_space)
    self.replay = make_replay(config, 'replay')
    self.stream = partial(make_stream, config)
    self.logger = make_logger(config) 

  def train(self, env, args):

    agent = self.agent
    replay = self.replay
    logger = self.logger

    logdir = elements.Path(args.logdir)
    step = logger.step
    usage = elements.Usage(**args.usage)
    train_agg = elements.Agg()
    epstats = elements.Agg()
    episodes = collections.defaultdict(elements.Agg)
    policy_fps = elements.FPS()
    train_fps = elements.FPS()

    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.LocalClock(args.log_every)
    should_report = embodied.LocalClock(args.report_every)
    should_save = embodied.LocalClock(args.save_every)

    @elements.timer.section('logfn')
    def logfn(tran, worker):
      episode = episodes[worker]
      tran['is_first'] and episode.reset()
      episode.add('score', tran['reward'], agg='sum')
      episode.add('length', 1, agg='sum')
      episode.add('rewards', tran['reward'], agg='stack')
      for key, value in tran.items():
        if value.dtype == np.uint8 and value.ndim == 3:
          if worker == 0:
            episode.add(f'policy_{key}', value, agg='stack')
        elif key.startswith('log/'):
          assert value.ndim == 0, (key, value.shape, value.dtype)
          episode.add(key + '/avg', value, agg='avg')
          episode.add(key + '/max', value, agg='max')
          episode.add(key + '/sum', value, agg='sum')
      if tran['is_last']:
        result = episode.result()
        logger.add({
            'score': result.pop('score'),
            'length': result.pop('length'),
        }, prefix='episode')
        rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

    # we are making environment externally so do not need this 
    # fns = [partial(self.make_env, i) for i in range(args.envs)] # function to make environment
    # driver = embodied.Driver(fns, parallel=not args.debug) # driver makes environment

    driver = embodied.Driver(env)
    driver.on_step(lambda tran, _: step.increment()) # adds stuff to do on driver step
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(logfn) 

    stream_train = iter(agent.stream(self.make_stream(replay, 'train')))
    stream_report = iter(agent.stream(self.make_stream(replay, 'report')))

    carry_train = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)

    def trainfn(tran, worker):
      if len(replay) < args.batch_size * args.batch_length:
        return
      for _ in range(should_train(step)):
        with elements.timer.section('stream_next'):
          batch = next(stream_train)
        carry_train[0], outs, mets = agent.train(carry_train[0], batch)
        train_fps.step(batch_steps)
        if 'replay' in outs:
          replay.update(outs['replay'])
        train_agg.add(mets, prefix='train')

    driver.on_step(trainfn)

    # CHECKPOINT SETUP / LOAD
    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.step = step
    cp.agent = agent
    cp.replay = replay
    if args.from_checkpoint:
      elements.checkpoint.load(args.from_checkpoint, dict(
          agent=partial(agent.load, regex=args.from_checkpoint_regex)))
    cp.load_or_save()

    print('Start training loop')
    policy = lambda *args: agent.policy(*args, mode='train')
    driver.reset(agent.init_policy) # sets carry to be init values 

    while step < args.steps: 

      driver(policy, steps=10) # run policy for 10 steps 
      if should_report(step) and len(replay):
        agg = elements.Agg()
        for _ in range(args.consec_report * args.report_batches):
          carry_report, mets = agent.report(carry_report, next(stream_report))
          agg.add(mets)
        logger.add(agg.result(), prefix='report')

      if should_log(step):
        logger.add(train_agg.result())
        logger.add(epstats.result(), prefix='epstats')
        logger.add(replay.stats(), prefix='replay')
        logger.add(usage.stats(), prefix='usage')
        logger.add({'fps/policy': policy_fps.result()})
        logger.add({'fps/train': train_fps.result()})
        logger.add({'timer': elements.timer.stats()['summary']})
        logger.write()

      if should_save(step):
        cp.save()

    logger.close()

"""
Utility functions

"""

def make_agent(config, obs_space, act_space):

  from repos.dreamerv3.dreamerv3.agent import Agent

  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in act_space.items() if k != 'reset'}

  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)

  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir

  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


# def make_env(config, index, **overrides):
#   suite, task = config.task.split('_', 1)
#   ctor = { # removed memmaze case
#       'dummy': 'embodied.envs.dummy:Dummy',
#       'gym': 'embodied.envs.from_gym:FromGym',
#       'dm': 'embodied.envs.from_dmenv:FromDM',
#       'crafter': 'embodied.envs.crafter:Crafter',
#       'dmc': 'embodied.envs.dmc:DMC',
#       'atari': 'embodied.envs.atari:Atari',
#       'atari100k': 'embodied.envs.atari:Atari',
#       'dmlab': 'embodied.envs.dmlab:DMLab',
#       'minecraft': 'embodied.envs.minecraft:Minecraft',
#       'loconav': 'embodied.envs.loconav:LocoNav',
#       'pinpad': 'embodied.envs.pinpad:PinPad',
#       'langroom': 'embodied.envs.langroom:LangRoom',
#       'procgen': 'embodied.envs.procgen:ProcGen',
#       'bsuite': 'embodied.envs.bsuite:BSuite'
#   }[suite]
#   if isinstance(ctor, str): # import and get constructor class
#     module, cls = ctor.split(':')
#     module = importlib.import_module(module)
#     ctor = getattr(module, cls) 
#   kwargs = config.env.get(suite, {}) 
#   kwargs.update(overrides)
#   if kwargs.pop('use_seed', False):
#     kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
#   if kwargs.pop('use_logdir', False):
#     kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
#   env = ctor(task, **kwargs)
#   return wrap_env(env, config)

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

def make_stream(config, replay, mode):
  fn = partial(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream
