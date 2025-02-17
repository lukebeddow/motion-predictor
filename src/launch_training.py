from datetime import datetime
import os
from time import sleep
from random import random
import hydra
import logging
import wandb
import sys

repo_path = os.path.abspath("repos/dreamerv3/")
if not os.path.exists(repo_path):
  print(f"Error: Path does not exist: {repo_path}")

sys.path.insert(0, repo_path)
import dreamerv3
import embodied

from trainers import dreamer_trainer
# from utils.logger import ProjectLogger

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

def create_savedir(cfg):
  """
  Create a new folder to save the run outputs in. If a folder with the same name
  exists, it adds progressively increasing integers on the end (_1, _2, etc)
  """

  # get the savedir requested
  folderpath = cfg.savedir

  # create the new folder
  if not os.path.exists(folderpath):
    os.makedirs(folderpath)
  else:
    # if the folder name already exists, add a distinguishing number
    i = 1
    new_foldername = "{0}_{1}"
    while os.path.exists(new_foldername.format(folderpath, i)):
      i += 1
    os.makedirs(new_foldername.format(folderpath, i))

    logging.warning(
        f"launch_training.create_savedir() warning: savedir requested '{folderpath}' already existed, changing savedir to '{new_foldername}'")

    # update the modified save location in the config
    cfg.savedir = new_foldername.format(folderpath, i)

  logging.info(
      f"launch_training.create_savedir(): savedir for this run created at: '{cfg.savedir}'")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

  starting_time = datetime.now()
  cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

  # with wandb.init(config=cfg_container, **cfg_container['wandb']):

  # initialise weights and biases
  wandb_instance = wandb.init(config=cfg_container, **cfg_container['wandb'])

  logging.info(cfg) # echo config
  create_savedir(cfg) # create the save location for this run

  # create a logging object for during the training
  logger = ProjectLogger(wandb_instance=wandb_instance, **cfg.logger)

  # create the environment from configuation settings
  env = hydra.utils.instantiate(cfg.env) 

  # build the agent / model from configuration settings 

  # define the input and output sizes for the network
  cfg.model.network.obs_dim = env.obs_dim
  cfg.model.network.act_dim = env.act_dim

  if cfg.model.name == 'ppo':

    #  create the network from configuration settings
    network = hydra.utils.instantiate(cfg.model.network)

    # create the agent from configuration settings
    agent = hydra.utils.instantiate(cfg.model.agent)
    agent.init(network)

    # create the trainer
    trainer = hydra.utils.instantiate(cfg.trainer, agent=agent, env=env, logger=logger)

    # save the full configuration, now that everything is ready
    OmegaConf.save(cfg, cfg.savedir + "/config.yaml")

    # now run the training
    trainer.train()

  elif cfg.model.name == 'dreamer': 

    # setup agent 
    agent = dreamer_trainer.DreamerTrainer(cfg, env)

    # start training
    agent.train(env)

  print_time_taken(starting_time)



if __name__ == "__main__":

  main()