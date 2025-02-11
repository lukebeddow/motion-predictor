from datetime import datetime
import argparse
from time import sleep
from random import random
import hydra
import logging
import wandb

from env.env import GymHandler
from agents import policy_gradient as pg
from trainers import ppo_trainer

from omegaconf import DictConfig, OmegaConf

def print_time_taken():

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
  print("\nStarted at:", starting_time.strftime(datestr))
  print("Finished at:", datetime.now().strftime(datestr))
  print(f"Time taken was {d[0]:.0f} days {h[0]:.0f} hrs {m[0]:.0f} mins {s:.0f} secs\n")

@hydra.main(config_path="/workspace/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

  cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
  with wandb.init(config=cfg_container, **cfg_container['wandb']):

    logging.info(cfg) # echo config
    env = GymHandler(cfg.env.name, cfg.env.seed)

    if cfg.model.name == 'ppo':
        
        # make the network for the agent
        layers = [128 for i in range(2)]
        network = pg.MLPActorCriticPG(env.obs_dim, env.act_dim, hidden_sizes=layers,
                                  continous_actions=True)
        
        # make the agent
        agent = pg.Agent_PPO(device=cfg.device, **cfg.model.algo)
        agent.init(network)

        # create the trainer
        trainer = ppo_trainer.Trainer(agent, env, **cfg.model.trainer)
        # run the training
        trainer.train()
        print_time_taken()

    else:
      raise RuntimeError(f"launch_trainig.py error: program name given = {cfg.model}, not recognised")



  # select environment, agent 


    # # starting time
    # starting_time = datetime.now()

    # # how to format dates/times
    # datestr = "%Y-%m-%d_%H-%M"

    # # define arguments and parse them
    # parser = argparse.ArgumentParser()

    # parser.add_argument("-j", "--job",          default=None, type=int) # job input number
    # parser.add_argument("-t", "--timestamp",    default=None)           # timestamp
    # parser.add_argument("-p", "--program",      default="default")      # program name to select from if..else if
    # parser.add_argument("-d", "--device",       default="cpu")          # override device
    # parser.add_argument("-r", "--render",       action="store_true")    # render window during training
    # parser.add_argument("--log-level",          type=int, default=1)    # set script log level
    # parser.add_argument("--rngseed",            default=None)           # turns on reproducible training with given seed (slower)

    # args = parser.parse_args()

    # # use a given timestamp, otherwise use the current time
    # timestamp = args.timestamp if args.timestamp else starting_time.strftime(datestr)

    # # echo key command line inputs
    # if args.log_level > 0:
    #   print("launch_training.py is preparing to train:")
    #   print(" -> Job number:", args.job)
    #   print(" -> Timestamp:", timestamp)
    #   print(" -> Program name:", args.program)
    #   print(" -> Device:", args.device)

    # # ----- now run the training program ----- #

if __name__ == "__main__":
  main()