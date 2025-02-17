from datetime import datetime
import os
from time import sleep
from random import random
import hydra
import logging
import wandb

import isaacgym
# from env.gym import GymHandler
# from trainers import base_trainer
# import agents.policy_gradient as pg
from utils.logger import ProjectLogger

import hydra 
import elements
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

    logging.warning(f"launch_training.create_savedir() warning: savedir requested '{folderpath}' already existed, changing savedir to '{new_foldername}'")
    
    # update the modified save location in the config
    cfg.savedir = new_foldername.format(folderpath, i)

  logging.info(f"launch_training.create_savedir(): savedir for this run created at: '{cfg.savedir}'")

def parkour(cfg, wandb_instance):
  """
  Debugging and developing integrating extreme-parkour into the repo. This function
  finishes with 'exit()' to kill the python process. The aim of this function is to
  test code without making main() dirty.
  """

  starting_time = datetime.now()

  from env.rsl_legged.rsl_legged import RSLEnv
  from trainers.roa_trainer import ROATrainer

  # create a logging object for during the training
  logger = ProjectLogger(wandb_instance=wandb_instance, **cfg.logger)

  # create the environment from configuation settings
  env = RSLEnv()

  # create trainer which wraps their on-policy runner
  trainer = ROATrainer(env=env, logger=logger, train_config=env.train_cfg_dict,
                       **cfg.trainer)
  # trainer = hydra.utils.instantiate(cfg.trainer, env=env, logger=logger,
  #                                   train_config=env.rsl_config)

  # perform the actual training
  trainer.train()

  print_time_taken(starting_time)
  exit()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

  # ----- initial setup ----- #

  starting_time = datetime.now()
  cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

  # initialise weights and biases
  wandb_instance = wandb.init(config=cfg_container, **cfg_container['wandb'])

  logging.info(cfg) # echo config
  create_savedir(cfg) # create the save location for this run

  import modelsaver
  ms = modelsaver.ModelSaver("run")
  ms.save("test.txt", txtstr="test file", txtonly=True)
  print("Current directory is:", os.getcwd())

  exit()

  # ----- check for special cases ----- #

  if cfg.exp_name == "parkour_dev": parkour(cfg, wandb_instance)

  # ----- pre-training setup ----- #

  # create a logging object for during the training
  logger = ProjectLogger(wandb_instance=wandb_instance, **cfg.logger)

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

  # create the trainer
  trainer = hydra.utils.instantiate(cfg.trainer, agent=agent, env=env, logger=logger)

  # save the full configuration, now that everything is ready
  OmegaConf.save(cfg, cfg.savedir + "/config.yaml")

  # ----- execute the training ----- #

  trainer.train()
  print_time_taken(starting_time)

if __name__ == "__main__":

  main()