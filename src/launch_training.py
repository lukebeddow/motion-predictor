from datetime import datetime
import argparse
from time import sleep
from random import random

from env.env import GymHandler
from trainer import Trainer
from agents.policy_gradient import Agent_PPO, MLPActorCriticPG

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

if __name__ == "__main__":

  # starting time
  starting_time = datetime.now()

  # how to format dates/times
  datestr = "%Y-%m-%d_%H-%M"

  # define arguments and parse them
  parser = argparse.ArgumentParser()

  parser.add_argument("-j", "--job",          default=None, type=int) # job input number
  parser.add_argument("-t", "--timestamp",    default=None)           # timestamp
  parser.add_argument("-p", "--program",      default="default")      # program name to select from if..else if
  parser.add_argument("-d", "--device",       default="cpu")          # override device
  parser.add_argument("-r", "--render",       action="store_true")    # render window during training
  parser.add_argument("--log-level",          type=int, default=1)    # set script log level
  parser.add_argument("--rngseed",            default=None)           # turns on reproducible training with given seed (slower)

  args = parser.parse_args()

  # use a given timestamp, otherwise use the current time
  timestamp = args.timestamp if args.timestamp else starting_time.strftime(datestr)

  # echo key command line inputs
  if args.log_level > 0:
    print("launch_training.py is preparing to train:")
    print(" -> Job number:", args.job)
    print(" -> Timestamp:", timestamp)
    print(" -> Program name:", args.program)
    print(" -> Device:", args.device)

  # ----- now run the training program ----- #

  if args.program == "default":

    # make the environment
    env = GymHandler("Pendulum-v1")

    # make the network for the agent
    layers = [128 for i in range(2)]
    network = MLPActorCriticPG(env.obs_dim, env.act_dim, hidden_sizes=layers,
                               continous_actions=True)
    
    # make the agent
    agent = Agent_PPO(device=args.device)
    agent.init(network)

    # create the trainer
    trainer = Trainer(agent, env, rngseed=args.rngseed, log_level=args.log_level,
                      device=args.device, render=args.render)

    # run the training
    trainer.train()

    print_time_taken()

  else:
    raise RuntimeError(f"launch_trainig.py error: program name given = {args.program}, not recognised")