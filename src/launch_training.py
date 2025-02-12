from datetime import datetime
import argparse
from time import sleep
from random import random
import hydra
import logging
import wandb

from env.gym import GymHandler
from trainers import base_trainer
import agents.policy_gradient as pg

import hydra 
from omegaconf import DictConfig, OmegaConf


def print_time_taken(starting_time):

  """
  Print the time taken since the training started.

  starting_time should be a global datetime object, from datetime.now()
  """

  finishing_time = datetime.now()
  time_taken = finishing_time - starting_time
  d = divmod(time_taken.total_seconds(), 86400)
  h = divmod(d[1], 3600)
  m = divmod(h[1], 60)
  s = m[1]
  print("\nStarted at:", starting_time.strftime("%Y-%m-%d_%H-%M"))
  print("Finished at:", datetime.now().strftime("%Y-%m-%d_%H-%M"))
  print(f"Time taken was {d[0]:.0f} days {h[0]:.0f} hrs {m[0]:.0f} mins {s:.0f} secs\n")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

  starting_time = datetime.now()
  cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

  with wandb.init(config=cfg_container, **cfg_container['wandb']):

    logging.info(cfg) # echo config

    # create the environment from configuation settings
    env = hydra.utils.instantiate(cfg.env)

    # define the input and output sizes for the network
    cfg.model.network.obs_dim = env.obs_dim
    cfg.model.network.act_dim = env.act_dim

    # create the network from configuration settings
    network = hydra.utils.instantiate(cfg.model.network)

    # create the agent from configuration settings
    agent = hydra.utils.instantiate(cfg.model.agent)
    agent.init(network)

    # create the trainer from configuration settings
    logger = hydra.utils.instantiate(cfg.logger)
    trainer = hydra.utils.instantiate(cfg.trainer, agent=agent, env=env, logger=logger)

    # now run the training
    trainer.train()
    print_time_taken(starting_time)

if __name__ == "__main__":

  main()