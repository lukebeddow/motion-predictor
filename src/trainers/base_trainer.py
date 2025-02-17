import os
import torch
import numpy as np
from dataclasses import dataclass, asdict
from itertools import count
import time
import random
from datetime import datetime
import functools
import wandb

from modelsaver import ModelSaver

class TrackTraining:

  def __init__(self, rolling_average_num=250, plt_frequency_seconds=1,
               use_wandb=False):
    """
    Track training data and generate matplotlib plots. To save test metrics pass
    in a list of metric names eg ["success_rate", "average_force", ...]. Metrics
    are stored as floats
    """
    # extract inputs
    self.wandb_enabled = use_wandb
    self.numpy_float = np.float32
    self.avg_num = rolling_average_num
    self.plt_frequency_seconds = plt_frequency_seconds
    self.last_log_time = time.process_time()
    # plotting options
    self.plot_raw = False # default is to plot rolling averages
    # general
    self.episodes_done = 0
    self.last_plot = 0
    self.per_action_time_taken = np.array([], dtype=self.numpy_float)
    self.avg_time_taken = np.array([], dtype=self.numpy_float)
    # training data
    self.train_data = {
        # automatically handled by this class
        "episodes_done" : 0,
        "episodes" : np.array([], dtype=np.int32),
        # user should log extra metrics with TrackTraining.log_episode(dict_of_name_value_pairs)
    }
    # testing data
    self.test_data = {
        # automatically handled by this class
        "tests_done" : 0,
        "test_episodes" : np.array([], dtype=np.int32),
        # user should log extra metrics with TrackTraining.log_test(dict_of_name_value_pairs)
    }
    self.base_metrics = (
        list(self.train_data.keys()) + list(self.test_data.keys()) + ["averages_done"]
    )
    # for plotting
    self.fig = None
    self.axs = None
    self.line_dict = {}

  def add_metrics(self, metric_names, type, dtypes=None, values=None):
    """
    Add additional metrics for logging.
      - metric_names: list of names associated with each metric
      - type: either 'train' or 'test', to indicate when metrics are logged
      - dtypes: optional list of every dtype per metric, should be numpy
      - values: optional list of initial values to save per metric
    """

    n = len(metric_names)

    if n == 0:
      print("TrackTraining.add_metrics() error: metric names should be a list of length > 0")

    if type not in ["train", "test"]:
      print(
          f"TrackTraining.add_metrics() error: type must be set to either 'train' or 'test', recieved '{type}'")

    if dtypes is None: 
      dtypes = [self.numpy_float for _ in range(n)]
    elif len(dtypes) != n:
      print(
          f"TrackTraining.add_metrics() error: number of metrics {n} != length of dtypes {len(dtypes)}. Either set dtypes=None, or add one for every metric (should be numpy types)")

    if values is None: 
      values = [None for _ in range(n)]
    elif len(values) != n:
      print(
          f"TrackTraining.add_metrics() error: number of metrics {n} != length of values {len(values)}. Either set values=None, or add one for every metric (can be None for some)")

    for i in range(n):
      vals = [] if values[i] == None else values[i]
      if metric_names[i] in self.base_metrics:
        print(
            f"TrackTraining.add_metrics() error: metric_name given '{metric_names[i]}' is not allowed, this name is reserved for the class internal usage")
      if type == "train":
        self.train_data[metric_names[i]] = np.array(vals, dtype=dtypes[i])
      elif type == "test":
        self.test_data[metric_names[i]] = np.array(vals, dtype=dtypes[i])

  def log_training_episode(self, log_dict):
    """
    Log one training episode
    """

    for key, value in log_dict.items():
      if key in self.train_data:
        self.train_data[key] = np.append(self.train_data[key], value)
      else:
        print(
            f"TrackTraining.log_training_episode() warning: log value with name {key} not found in self.train_data. "
            "Not logged - metrics should be added first with TrackTraining.add_metrics()"
            )

    self.train_data["episodes_done"] += 1
    self.train_data["episodes"] = np.append(
        self.train_data["episodes"], self.train_data["episodes_done"])

    # update average information
    new_data = self.calc_static_average()

    if self.wandb_enabled and new_data:
      # temporary test of wanbd
      wandb.log({'reward':self.train_data['reward'][-1]})

  def log_test(self, log_dict):
    """
    Log one test
    """

    for key, value in log_dict.items():
      if key in self.test_data:
        self.test_data[key] = np.append(self.test_data[key], value)
      else:
        print(
            f"TrackTraining.log_test() warning: log value with name {key} not found in self.test_data. Not logged - metrics should be added first with TrackTraining.add_metrics()")

    self.test_data["tests_done"] += 1
    self.test_data["test_episodes"] = np.append(
        self.test_data["test_episodes"], self.train_data["episodes_done"])

  def calc_static_average(self):
    """
    Average rewards and durations to reduce data points
    """

    if not hasattr(self, "train_data_avg"):
      self.train_data_avg = {}
      for key, value in self.train_data.items():
        if key == "episodes_done":
          self.train_data_avg["averages_done"] = 0
        else:
          self.train_data_avg[key] = np.array([], dtype=value.dtype)

    # find number of points we can average
    num_points_averaged = self.train_data_avg["averages_done"] * self.avg_num
    num_points_unaveraged = self.train_data["episodes_done"] - num_points_averaged
    num_new_averages = num_points_unaveraged // self.avg_num

    # if we points which have not been averaged yet
    if num_new_averages > 0:

      for key, value in self.train_data_avg.items():

        if key == "averages_done": continue

        unaveraged_points = self.train_data[key][num_points_averaged:]
        assert(len(unaveraged_points) // self.avg_num == num_new_averages)

        for i in range(num_new_averages):
          new_average = np.mean(
              unaveraged_points[i * self.avg_num : (i + 1) * self.avg_num])
          self.train_data_avg[key] = np.append(self.train_data_avg[key], new_average)

      self.train_data_avg["averages_done"] += num_new_averages

    return bool(num_new_averages)

  def plot_matplotlib(self, xdata, ydata, ylabel, title, axs, label=None):
    """
    Plot onto a given axs
    """
    if title in self.line_dict:
      self.line_dict[title].set_xdata(xdata)
      self.line_dict[title].set_ydata(ydata)
      axs.relim()
      axs.autoscale_view()
    else:
      self.line_dict[title], = axs.plot(xdata, ydata, label=label)
      axs.set_title(title, fontstyle="italic")
      if not self.plot_raw: ylabel = f"{ylabel} (avg={self.avg_num})"
      axs.set(ylabel=ylabel)

    # axs.plot(xdata, ydata, label=label)
    # axs.set_title(title, fontstyle="italic")
    # axs.set(ylabel=ylabel)

  def plot(self, plt_frequency_seconds=None):
    """
    Plot training results figures, pass a frequency to plot only if enough
    time has elapsed
    """

    if self.train_data["episodes_done"] < 5: return

    if plt_frequency_seconds is None:
      plt_frequency_seconds = self.plt_frequency_seconds

    # if not enough time has elapsed since the last plot
    if (self.last_plot + plt_frequency_seconds > time.time()):
      return

    if not "plt" in globals():
      global plt, display
      import matplotlib.pyplot as plt
      from IPython import display
      plt.ion()

    num = len(self.train_data) + len(self.test_data) - len(self.base_metrics) + 1

    # calculate the number of rows and columns needed for plotting
    rows = int(np.floor(np.sqrt(num)))
    cols = int(np.ceil(num / rows))

    if self.fig == None:
      self.fig, self.axs = plt.subplots(cols, rows)
      self.fig.set_size_inches(rows * 2, cols * 2)
      self.fig.tight_layout()
      display.display(self.fig, display_id="live_plot")

      if len(self.axs.shape) == 1:
        self.axs = self.axs.reshape((self.axs.shape[0], 1))

    ind_row = 0
    ind_col = 0

    if self.plot_raw:
      train_metrics = self.train_data.items()
      train_xdata = np.array(self.train_data["episodes"])
    else:
      train_metrics = self.train_data_avg.items()
      train_xdata = np.array(self.train_data_avg["episodes"])

    for key, value in train_metrics:

      if key in self.base_metrics: continue

      self.plot_matplotlib(train_xdata, value, key, key, self.axs[ind_col][ind_row])

      ind_row += 1
      if ind_row >= rows:
        ind_row = 0
        ind_col += 1

    for key, value in self.test_data.items():

      if key in self.base_metrics: continue

      self.plot_matplotlib(
          np.array(self.test_data["test_episodes"]), value, key, key, self.axs[ind_col][ind_row])

      ind_row += 1
      if ind_row >= rows:
        ind_row = 0
        ind_col += 1

    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

    # for juypter notebook
    display.update_display(self.fig, display_id="live_plot")

    # # for normal windows
    # plt.pause(0.001)

    # save that we plotted
    self.last_plot = time.time()

    return

  def print_train_metrics(self, avg_only=True):
    if not avg_only or self.train_data["episodes_done"] % self.avg_num == 0:
      print(f"Episode {self.train_data['episodes_done']} training metrics:")
      for key, value in self.train_data.items():
        if key not in self.base_metrics:
          print(f" -> {key} = {value[-1]:.3f}")
      print()

  def print_test_metrics(self):
    print(
        f"Test {self.test_data['tests_done']} metrics, at training episode {self.train_data['episodes_done']}:")
    for key, value in self.test_data.items():
      if key not in self.base_metrics:
        print(f" -> {key} = {value[-1]:.3f}")
    print()

class BaseTrainer:

  @dataclass
  class Parameters:

    num_episodes: int = 1_000
    test_freq: int = 200
    save_freq: int = 200
    use_curriculum: bool = False

  def __init__(self, agent, env, logger, num_episodes:int,
               test_freq:int, save_freq:int, use_curriculum:bool,
               seed=None, device="cpu", log_level=1, plot=False,
               render=False, group_name="", run_name="run",
               save=True, savedir="run", episode_log_rate=10, strict_seed=False):
    """
    Class that trains agents in an environment
    """

    self.params = LegacyTrainer.Parameters()
    self.logger = logger

    self.agent = agent
    self.env = env

    self.saved_trainer_params = False
    self.last_loaded_agent_id = None
    self.last_saved_agent_id = None
    self.episode_fcn = None

    self.params.num_episodes = num_episodes
    self.params.test_freq = test_freq
    self.params.save_freq = save_freq
    self.params.use_curriculum = use_curriculum

    # input class options
    self.rngseed = seed
    self.device = device
    self.log_level = log_level
    self.plot = plot
    self.render = render
    self.log_rate_for_episodes = episode_log_rate

    # set up saving
    self.setup_saving(run_name, group_name, savedir, enable_saving=save)

    # are we plotting
    if self.plot:
      global plt, display
      import matplotlib.pyplot as plt
      from IPython import display
      plt.ion()

    # seed the environment (skip if given None for agent and env)
    # training only reproducible if torch.manual_seed() set BEFORE agent network initialisation
    self.training_reproducible = strict_seed
    if agent is not None and env is not None: self.seed(strict=strict_seed)
    else:
      if strict_seed or seed is not None:
        raise RuntimeError(
            "Trainer.__init__() error: agent and/or env is None, environment is not seeded by rngseed or strict_seed was set")
      elif self.log_level >= 2:
        print("Trainer.__init__() warning: agent and/or env is None and environment is NOT seeded")

    if self.log_level > 0:
      print("Trainer settings:")
      print(" -> Run name:", self.run_name)
      print(" -> Group name:", self.group_name)
      print(" -> Given seed:", seed)
      print(" -> Training reproducible:", self.training_reproducible)
      print(" -> Using device:", self.device)
      print(" -> Save enabled:", self.enable_saving)
      if self.enable_saving:
        print(" -> Save path:", self.modelsaver.path)

  def setup_saving(self, run_name="run", group_name="",
                   savedir="./", enable_saving=None):
    """
    Provide saving information and enable saving of models during training. The
    save() function will not work without first running this function
    """

    if enable_saving is not None:
      self.enable_saving = enable_saving

    # save information and create modelsaver to manage saving/loading
    self.group_name = group_name
    self.run_name = run_name
    self.savedir = savedir

    if len(self.savedir) > 0 and self.savedir[-1] != "/": self.savedir += "/"

    if self.enable_saving:
      self.modelsaver = ModelSaver(self.savedir, log_level=self.log_level)

  def to_torch(self, data, dtype=torch.float32):
    if torch.is_tensor(data):
      return data.unsqueeze(0)
    else:
      return torch.tensor(data, device=self.device, dtype=dtype).unsqueeze(0)

  def seed(self, rngseed=None, strict=None):
    """
    Set a random seed for the entire environment
    """
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: 
        self.training_reproducible = False
        rngseed = np.random.randint(0, 2_147_483_647)

    torch.manual_seed(rngseed)
    self.rngseed = rngseed
    self.agent.seed(rngseed)
    self.env.seed(rngseed)

    # if we want to ensure reproducitibilty at the cost of performance
    if strict is None: strict = self.training_reproducible
    if strict:
      # increases GPU usage by 24MiB, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility and ctrl+f "CUBLAS_WORKSPACE_CONFIG"
      os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
      torch.backends.cudnn.benchmark = False
      torch.use_deterministic_algorithms(mode=True)
    else: self.training_reproducible = False

  def save(self, extra_text_file_name=None, extra_text_file_string=None):
    """
    Save the state of the current trainer and agent.

    To save an additional text file with the agent set:
      * txtfilename: str -> name of additional file to save, {agent_file_name}_{txtfilestr}.txt
      * txtfilestr: str -> content of additional file
    """

    if not self.enable_saving: 
      if self.log_level > 1:
        print("Trainer.save(): self.enable_saving = False, nothing saved")
      return

    # save the actual (saves a new file each time, numbering 1,2,3,...)
    self.modelsaver.save(self.agent.name, pyobj=self.agent.get_save_state(),
                         txtstr=extra_text_file_string, txtlabel=extra_text_file_name)

    self.last_saved_agent_id = self.modelsaver.last_saved_id

  def get_save_id(self, episode):
    """
    Return the save id associated with a given episode. Note: if the test_freq
    or save_freq is changed, this function will no longer output correct ids
    """
    first_save = 1
    if self.params.test_freq == self.params.save_freq:
      save_id = first_save + (episode // self.params.test_freq)
    else:
      save_id = first_save + (episode // self.params.test_freq
                              + episode // self.params.save_freq
                              - episode // (np.lcm(self.params.test_freq, 
                                                   self.params.save_freq)))
    return save_id

  def get_param_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
        "rngseed" : self.rngseed,
        "training_reproducible" : self.training_reproducible,
        "saving_enabled" : self.enable_saving,
    })
    return param_dict


  def load(self, run_name, id=None, group_name=None, path_to_run_folder=None, 
           agentonly=False, trackonly=False):
    """
    Load a model given a path to it
    """

    if agentonly and trackonly:
      raise RuntimeError(
          "Trainer.load() error: agentonly=True and trackonly=True, incompatible arguments")

    # check if modelsaver is defined
    if not hasattr(self, "modelsaver"):
      if path_to_run_folder is not None:
        print(
            f"load not given a modelsaver, making one from path_to_group: {path_to_run_folder}")
        self.modelsaver = ModelSaver(path_to_run_folder, log_level=self.log_level)
      elif group_name is not None:
        # try to find the group from this folder
        pathhere = os.path.dirname(os.path.abspath(__file__))
        print(
            f"load not given modelsaver or path_to_group, assuming group is local at {pathhere + '/' + self.savedir}")
        self.modelsaver = ModelSaver(pathhere + "/" + self.savedir + "/" + self.group_name,
                                     log_level=self.log_level)
      else:
        raise RuntimeError(
            "load not given a modelsaver and either of a) path_to_run_folder b) group_name (if group can be found locally)")

    # don't do this now, we change to save in self.savedir, no run folder
    # # enter the run folder (exit if already in one)
    # self.modelsaver.enter_folder(run_name)

    # folderpath ignores the current folder, so add that if necessary
    if path_to_run_folder is not None:
      path_to_run_folder += "/" + run_name

    load_agent = self.modelsaver.load(id=id, folderpath=path_to_run_folder,
                                      filenamestarts="Agent")
    self.last_loaded_agent_id = self.modelsaver.last_loaded_id

    # get the name of the agent from the filename saved with
    name = self.modelsaver.get_recent_file(name="Agent")
    name = name.split("/")[-1]

    # trim out the agent part
    name = name[:len("Agent") + len(self.modelsaver.file_ext())]

    # do we have the agent already, if not, create it
    if self.agent is None:
      to_exec = f"""self.agent = {name}()"""
      exec(to_exec)
    # try to load the save state, but catch situation where we have empty agent
    try:
      self.agent.load_save_state(load_agent, device=self.device)
    except NotImplementedError as e:
      # agent is not actually loaded, so load as above
      to_exec = f"""self.agent = {name}()"""
      exec(to_exec)
      self.agent.load_save_state(load_agent, device=self.device)
    if self.log_level >= 2 and hasattr(self.agent, "debug"): 
      self.agent.debug = True

    # reseed - be aware this will not be contingous
    self.training_reproducible = False # training no longer reproducible
    self.seed()

  def run_episode(self, i_episode, test=False):
    """
    Run one episode of RL
    """

    # initialise environment and state
    obs = self.env.reset()
    obs = self.to_torch(obs)

    ep_start = time.time()

    cumulative_reward = 0

    # count up through actions
    for t in count():

      if self.log_level >= 3: print("Episode", i_episode, "action", t)

      # select and perform an action
      action = self.agent.select_action(obs, decay_num=i_episode, test=test)
      (new_obs, reward, terminated, truncated, info) = self.env.step(action)

      # render the new environment
      if self.render: self.env.render()

      if terminated or truncated: done = True
      else: done = False

      # convert data to torch tensors on specified device
      new_obs = self.to_torch(new_obs)
      reward = self.to_torch(reward)
      action = action.to(self.device).unsqueeze(0) # from Tensor([x]) -> Tensor([[x]])
      truncated = self.to_torch(truncated, dtype=torch.bool)

      # store if it was a terminal state (ie either terminated or truncated)
      done = self.to_torch(done, dtype=torch.bool)

      # perform one step of the optimisation on the policy network
      if not test:
        self.agent.update_step(obs, action, new_obs, reward, done, truncated)

      obs = new_obs
      cumulative_reward += reward.cpu()

      # allow end of episode function to be set by the user
      if self.episode_fcn is not None: self.episode_fcn()

      # check if this episode is over and log if we aren't testing
      if done:

        ep_end = time.time()
        time_per_step = (ep_end - ep_start) / float(t + 1)

        if self.log_level >= 3:
          print(f"Time for episode was {ep_end - ep_start:.3f}s"
                f", time per action was {time_per_step * 1e3:.3f} ms")

        # if we are testing, no data is logged
        if test: break

        # save training data
        self.logger.scalar("reward", cumulative_reward[0])
        self.logger.scalar("duration", t + 1)
        self.logger.log_step()

        cumulative_reward = 0

        break

  def train(self, i_start=None, num_episodes_abs=None, num_episodes_extra=None):
    """
    Run a training
    """

    if i_start is None:
      i_start = int(self.logger.step)

    if num_episodes_abs is not None:
      self.params.num_episodes = num_episodes_abs

    if num_episodes_extra is not None:
      self.params.num_episodes = i_start + num_episodes_extra

    if num_episodes_abs is not None and num_episodes_extra is not None:
      if self.log_level > 0:
        print(
            f"Trainer.train() warning: num_episodes={num_episodes_abs} (ignored) and num_episodes_extra={num_episodes_extra} (used) were both set. Training endpoing set as {self.params.num_episodes}")

    if i_start >= self.params.num_episodes:
      raise RuntimeError(
          f"Trainer.train() error: training episode start = {i_start} is greater or equal to the target number of episodes = {self.params.num_episodes}")

    # if this is a fresh, new training
    if i_start == 0:
      # save starting network parameters and training settings
      self.save()


    if self.log_level > 0:
      print(
          f"\nBegin training, target is {self.params.num_episodes} episodes\n", flush=True)

    # prepare the agent for training
    self.agent.set_device(self.device)
    self.agent.training_mode()

    # begin training episodes
    for i_episode in range(i_start + 1, self.params.num_episodes + 1):

      if self.log_level == 1 and (i_episode - 1) % self.log_rate_for_episodes == 0:
        print(f"Begin training episode {i_episode}", flush=True)
      elif self.log_level > 1:
        print(
            f"Begin training episode {i_episode} at {datetime.now().strftime('%H:%M')}", flush=True)

      self.run_episode(i_episode)

      # not implemented with new logger
      # # plot graphs to the screen
      # if self.plot: self.logger.plot()

      # check if we need to do any episode level updates (eg target network)
      self.agent.update_episode(i_episode)

      # save the network
      if i_episode % self.params.save_freq == 0 and i_episode != 0:
        self.save()

      # test the target network
      if i_episode % self.params.test_freq == 0 and i_episode != 0:
        self.test() # this function should save test data as well

    # the agent may require final updating at the end
    self.agent.update_episode(i_episode, finished=True)

    # save, log and plot now we are finished
    if self.log_level > 0:
      print("\nTraining complete, finished", i_episode, "episodes\n")

    # wrap up
    if self.render: self.env.render()

    # not implemented with new logger
    # if self.plot: self.logger.plot()

  # need to define these functions for Trainer base class

  def test(self):
    """
    Empty test function, should be overriden for each environment.

    Outline:

    self.env.start_test() # if env needs to do extra logging
    self.agent.testing_mode() # if using pytorch, don't forget .eval()

    for i_episode in range(num_trials):
      self.run_episode(i_episode, test=True)

    test_data = self.env.get_test_data()

    self.env.end_test() # disable any test specific behaviour
    self.agent.training_mode() # disable any test time changes

    test_data = self.process_test_data(test_data) # maybe post-process
    self.save_test(test_data) # maybe save the results

    # return the test data
    return test_data
    """
    pass


class LegacyTrainer:

  @dataclass
  class Parameters:

    num_episodes: int = 1_000
    test_freq: int = 200
    save_freq: int = 200
    use_curriculum: bool = False

  def __init__(self, agent, env, logger, num_episodes:int,
               test_freq:int, save_freq:int, use_curriculum:bool,
               seed=None, device="cpu", log_level=1, plot=False,
               render=False, group_name="", run_name="run",
               save=True, savedir="", episode_log_rate=10, strict_seed=False):
    """
    Class that trains agents in an environment
    """

    self.params = LegacyTrainer.Parameters()
    self.track = logger

    self.agent = agent
    self.env = env

    self.saved_trainer_params = False
    self.last_loaded_agent_id = None
    self.last_saved_agent_id = None
    self.episode_fcn = None

    self.params.num_episodes = num_episodes
    self.params.test_freq = test_freq
    self.params.save_freq = save_freq
    self.params.use_curriculum = use_curriculum

    # input class options
    self.rngseed = seed
    self.device = device
    self.log_level = log_level
    self.plot = plot
    self.render = render
    self.log_rate_for_episodes = episode_log_rate

    # set up logging
    train_metrics = ["duration", "reward"]
    self.track.add_metrics(train_metrics, "train")

    # set up saving
    self.train_param_savename = "Trainer_params"
    self.track_savename = "Tracking_info"
    self.setup_saving(run_name, group_name, savedir, enable_saving=save)

    # are we plotting
    if self.plot:
      global plt, display
      import matplotlib.pyplot as plt
      from IPython import display
      plt.ion()

    # seed the environment (skip if given None for agent and env)
    # training only reproducible if torch.manual_seed() set BEFORE agent network initialisation
    self.training_reproducible = strict_seed
    if agent is not None and env is not None: self.seed(strict=strict_seed)
    else:
      if strict_seed or seed is not None:
        raise RuntimeError(
            "Trainer.__init__() error: agent and/or env is None, environment is not seeded by rngseed or strict_seed was set")
      elif self.log_level >= 2:
        print("Trainer.__init__() warning: agent and/or env is None and environment is NOT seeded")

    if self.log_level > 0:
      print("Trainer settings:")
      print(" -> Run name:", self.run_name)
      print(" -> Group name:", self.group_name)
      print(" -> Given seed:", seed)
      print(" -> Training reproducible:", self.training_reproducible)
      print(" -> Using device:", self.device)
      print(" -> Save enabled:", self.enable_saving)
      if self.enable_saving:
        print(" -> Save path:", self.modelsaver.path)

  def setup_saving(self, run_name="run", group_name="",
                   savedir="", enable_saving=None):
    """
    Provide saving information and enable saving of models during training. The
    save() function will not work without first running this function
    """

    if enable_saving is not None:
      self.enable_saving = enable_saving

    # save information and create modelsaver to manage saving/loading
    self.group_name = group_name
    self.run_name = run_name
    self.savedir = savedir

    if len(self.savedir) > 0 and self.savedir[-1] != "/": self.savedir += "/"

    if self.enable_saving:
      self.modelsaver = ModelSaver(self.savedir + self.group_name,
                                   log_level=self.log_level)

  def to_torch(self, data, dtype=torch.float32):
    if torch.is_tensor(data):
      return data.unsqueeze(0)
    else:
      return torch.tensor(data, device=self.device, dtype=dtype).unsqueeze(0)

  def seed(self, rngseed=None, strict=None):
    """
    Set a random seed for the entire environment
    """
    if rngseed is None:
      if self.rngseed is not None: rngseed = self.rngseed
      else: 
        self.training_reproducible = False
        rngseed = np.random.randint(0, 2_147_483_647)

    torch.manual_seed(rngseed)
    self.rngseed = rngseed
    self.agent.seed(rngseed)
    self.env.seed(rngseed)

    # if we want to ensure reproducitibilty at the cost of performance
    if strict is None: strict = self.training_reproducible
    if strict:
      # increases GPU usage by 24MiB, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility and ctrl+f "CUBLAS_WORKSPACE_CONFIG"
      os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
      torch.backends.cudnn.benchmark = False
      torch.use_deterministic_algorithms(mode=True)
    else: self.training_reproducible = False

  def save(self, txtfilename=None, txtfilestr=None, force_train_params=False,
           force_save_number=None):
    """
    Save the state of the current trainer and agent.

    To save an additional text file with the agent set:
      * txtfilename: str -> name of additional file to save, {agent_file_name}_{txtfilestr}.txt
      * txtfilestr: str -> content of additional file
    """

    if not self.enable_saving: 
      if self.log_level > 1:
        print("Trainer.save(): self.enable_saving = False, nothing saved")
      return

    if not self.modelsaver.in_folder:
      self.modelsaver.new_folder(name=self.run_name, notimestamp=True)

    # have we saved key information about the trainer
    if not self.saved_trainer_params or force_train_params:
      trainer_save = {
          "parameters" : self.params,
          "rngseed" : self.rngseed,
          "run_name" : self.run_name,
          "group_name" : self.group_name,
          "agent_name" : self.agent.name,
          "env_data" : self.env.get_save_state(),
      }

      # save trainer information (only once at the start of training)
      self.modelsaver.save(self.train_param_savename, pyobj=trainer_save)
      self.saved_trainer_params = True

    # determine what save_id to use
    save_id = self.get_save_id(self.track.episodes_done)
    save_id_agent = save_id if force_save_number is None else force_save_number

    # save tracking information (this file gets overwritten during training)
    self.modelsaver.save(self.track_savename, pyobj=self.track, 
                         suffix_numbering=False)

    # save the actual agent (saves a new file each time, numbering 1,2,3,...)
    self.modelsaver.save(self.agent.name, pyobj=self.agent.get_save_state(),
                         txtstr=txtfilestr, txtlabel=txtfilename,
                         force_suffix=save_id_agent)

    self.last_saved_agent_id = self.modelsaver.last_saved_id

  def get_save_id(self, episode):
    """
    Return the save id associated with a given episode. Note: if the test_freq
    or save_freq is changed, this function will no longer output correct ids
    """
    first_save = 1
    if self.params.test_freq == self.params.save_freq:
      save_id = first_save + (episode // self.params.test_freq)
    else:
      save_id = first_save + (episode // self.params.test_freq
                              + episode // self.params.save_freq
                              - episode // (np.lcm(self.params.test_freq, 
                                                   self.params.save_freq)))
    return save_id

  def get_param_dict(self):
    """
    Return a dictionary of hyperparameters
    """
    param_dict = asdict(self.params)
    param_dict.update({
        "rngseed" : self.rngseed,
        "training_reproducible" : self.training_reproducible,
        "saving_enabled" : self.enable_saving,
    })
    return param_dict

  def save_hyperparameters(self, filename="hyperparameters", strheader=None, 
                           print_terminal=None):
    """
    Save the model hyperparameters
    """

    if print_terminal is None:
      if self.log_level > 0: print_terminal = True
      else: print_terminal = False

    hyper_str = """"""
    if strheader is not None: hyper_str += strheader + "\n"

    hyper_str += "Trainer hyperparameters:\n\n"
    hyper_str += str(self.get_param_dict()).replace(",", "\n") + "\n\n"

    hyper_str += "Agent hyperparameters:\n\n"
    hyper_str += str(self.agent.get_params_dict()).replace(",", "\n") + "\n\n"

    hyper_str += "Env hyperparameters:\n\n"
    hyper_str += str(self.env.get_params_dict()).replace(",", "\n") + "\n\n"

    if print_terminal: print(hyper_str)

    if self.enable_saving:
      self.modelsaver.save(filename, txtstr=hyper_str, txtonly=True)

  def load(self, run_name, id=None, group_name=None, path_to_run_folder=None, 
           agentonly=False, trackonly=False):
    """
    Load a model given a path to it
    """

    if agentonly and trackonly:
      raise RuntimeError(
          "Trainer.load() error: agentonly=True and trackonly=True, incompatible arguments")

    # check if modelsaver is defined
    if not hasattr(self, "modelsaver"):
      if path_to_run_folder is not None:
        print(
            f"load not given a modelsaver, making one from path_to_group: {path_to_run_folder}")
        self.modelsaver = ModelSaver(path_to_run_folder, log_level=self.log_level)
      elif group_name is not None:
        # try to find the group from this folder
        pathhere = os.path.dirname(os.path.abspath(__file__))
        print(
            f"load not given modelsaver or path_to_group, assuming group is local at {pathhere + '/' + self.savedir}")
        self.modelsaver = ModelSaver(pathhere + "/" + self.savedir + "/" + self.group_name,
                                     log_level=self.log_level)
      else:
        raise RuntimeError(
            "load not given a modelsaver and either of a) path_to_run_folder b) group_name (if group can be found locally)")

    # enter the run folder (exit if already in one)
    self.modelsaver.enter_folder(run_name)

    # folderpath ignores the current folder, so add that if necessary
    if path_to_run_folder is not None:
      path_to_run_folder += "/" + run_name

    do_load_agent = not trackonly
    do_load_track = not agentonly
    do_load_training = not (trackonly + agentonly)

    if do_load_agent:

      load_agent = self.modelsaver.load(id=id, folderpath=path_to_run_folder,
                                        filenamestarts="Agent")
      self.last_loaded_agent_id = self.modelsaver.last_loaded_id

      # if we are only loading the agent
      if agentonly:
        if self.log_level > 0:
          print("Trainer.load() warning: AGENTONLY=TRUE, setting self.env=None for safety")
        self.env = None

        # get the name of the agent from the filename saved with
        name = self.modelsaver.get_recent_file(name="Agent")
        name = name.split("/")[-1]

        # trim out the agent part
        name = name[:5 + len(self.modelsaver.file_ext())]

        to_exec = f"""self.agent = {name}()"""
        exec(to_exec)
        self.agent.load_save_state(load_agent, device=self.device)

        if self.log_level > 0:
          print("Trainer.load(): AGENTONLY=True load now completed")

    if do_load_track:
      self.track = self.modelsaver.load(id=id, folderpath=path_to_run_folder, 
                                        filenamestarts=self.track_savename,
                                        suffix_numbering=self.trackinfo_numbering)
      if trackonly:
        if self.log_level > 0:
          print("Trainer.load() warning: TRACKONLY=TRUE, setting self.agent=None and self.env=None for safety")
        self.env = None
        self.agent = None

    if do_load_training:

      load_train = self.modelsaver.load(folderpath=path_to_run_folder,
                                        filenamestarts=self.train_param_savename)

      # extract loaded data
      self.params = load_train["parameters"]
      self.run_name = load_train["run_name"]
      self.group_name = load_train["group_name"]
      self.env.load_save_state(load_train["env_data"], device=self.device)

      # unpack curriculum information
      self.curriculum_dict = load_train["curriculum_dict"]
      if len(self.track.train_curriculum_stages) > 0:
        self.curriculum_dict["stage"] = self.track.train_curriculum_stages[-1]
      else: self.curriculum_dict["stage"] = 0

      # load in the curriculum functions
      if "curriculum_change_fcn" in load_train.keys():
        self.curriculum_change_fcn = load_train["curriculum_change_fcn"]
      if "curriculum_fcn" in load_train.keys():
        self.curriculum_fcn = load_train["curriculum_fcn"]
      else:
        # TEMPORARY FIX for program: mat_liftonly
        if self.group_name == "22-03-24":
          print("TEMPORARY CURRICULUM FIX: set curriculum function as curriculum_change_object_noise")
          from launch_training import curriculum_fcn_MAT
          self.curriculum_fcn = functools.partial(curriculum_fcn_MAT, self)
          print("CURRICULUM ACTIVE: Calling curriculum_fcn_MAT now")
          self.curriculum_fcn(10) # apply settings based on best success rate
        else:
          print("CURRICULUM WARNING: self.curriculum_fcn not set in dict")
          # apply initial stage settings
          self.curriculum_change(self.curriculum_dict["stage"])

      # run the curriculum function to update to the current curriculum
      if self.params.use_curriculum:
        self.curriculum_fcn(self.track.episodes_done)

      # do we have the agent already, if not, create it
      if self.agent is None:
        to_exec = f"""self.agent = {load_train["agent_name"]}()"""
        exec(to_exec)
      # try to load the save state, but catch situation where we have empty agent
      try:
        self.agent.load_save_state(load_agent, device=self.device)
      except NotImplementedError as e:
        # agent is not actually loaded, so load as above
        to_exec = f"""self.agent = {load_train["agent_name"]}()"""
        exec(to_exec)
        self.agent.load_save_state(load_agent, device=self.device)
      if self.log_level >= 2 and hasattr(self.agent, "debug"): 
        self.agent.debug = True

      # reseed - be aware this will not be contingous
      self.rngseed = load_train["rngseed"]
      self.training_reproducible = False # training no longer reproducible
      self.seed()

  def run_episode(self, i_episode, test=False):
    """
    Run one episode of RL
    """

    # initialise environment and state
    obs = self.env.reset()
    obs = self.to_torch(obs)

    ep_start = time.time()

    cumulative_reward = 0

    # count up through actions
    for t in count():

      if self.log_level >= 3: print("Episode", i_episode, "action", t)

      # select and perform an action
      action = self.agent.select_action(obs, decay_num=i_episode, test=test)
      (new_obs, reward, terminated, truncated, info) = self.env.step(action)

      # render the new environment
      if self.render: self.env.render()

      if terminated or truncated: done = True
      else: done = False

      # convert data to torch tensors on specified device
      new_obs = self.to_torch(new_obs)
      reward = self.to_torch(reward)
      action = action.to(self.device).unsqueeze(0) # from Tensor([x]) -> Tensor([[x]])
      truncated = self.to_torch(truncated, dtype=torch.bool)

      # store if it was a terminal state (ie either terminated or truncated)
      done = self.to_torch(done, dtype=torch.bool)

      # perform one step of the optimisation on the policy network
      if not test:
        self.agent.update_step(obs, action, new_obs, reward, done, truncated)

      obs = new_obs
      cumulative_reward += reward.cpu()

      # allow end of episode function to be set by the user
      if self.episode_fcn is not None: self.episode_fcn()

      # check if this episode is over and log if we aren't testing
      if done:

        ep_end = time.time()
        time_per_step = (ep_end - ep_start) / float(t + 1)

        if self.log_level >= 3:
          print(f"Time for episode was {ep_end - ep_start:.3f}s"
                f", time per action was {time_per_step * 1e3:.3f} ms")

        # if we are testing, no data is logged
        if test: break

        # save training data
        self.track.log_training_episode({
            "duration" : t + 1,
            "reward" : cumulative_reward,
        })

        cumulative_reward = 0

        break

  def train(self, i_start=None, num_episodes_abs=None, num_episodes_extra=None):
    """
    Run a training
    """

    if i_start is None:
      i_start = self.track.episodes_done

    if num_episodes_abs is not None:
      self.params.num_episodes = num_episodes_abs

    if num_episodes_extra is not None:
      self.params.num_episodes = i_start + num_episodes_extra

    if num_episodes_abs is not None and num_episodes_extra is not None:
      if self.log_level > 0:
        print(
            f"Trainer.train() warning: num_episodes={num_episodes_abs} (ignored) and num_episodes_extra={num_episodes_extra} (used) were both set. Training endpoing set as {self.params.num_episodes}")

    if i_start >= self.params.num_episodes:
      raise RuntimeError(
          f"Trainer.train() error: training episode start = {i_start} is greater or equal to the target number of episodes = {self.params.num_episodes}")

    # if this is a fresh, new training
    if i_start == 0:
      # save starting network parameters and training settings
      self.save()
      self.save_hyperparameters()
    else:
      # save a record of the training restart
      continue_label = f"Training is continuing from episode {i_start} with these hyperparameters\n"
      hypername = f"hyperparameters_from_ep_{i_start}"
      self.save_hyperparameters(filename=hypername, strheader=continue_label)

    if self.log_level > 0:
      print(
          f"\nBegin training, target is {self.params.num_episodes} episodes\n", flush=True)

    # prepare the agent for training
    self.agent.set_device(self.device)
    self.agent.training_mode()

    # begin training episodes
    for i_episode in range(i_start + 1, self.params.num_episodes + 1):

      if self.log_level == 1 and (i_episode - 1) % self.log_rate_for_episodes == 0:
        print(f"Begin training episode {i_episode}", flush=True)
      elif self.log_level > 1:
        print(
            f"Begin training episode {i_episode} at {datetime.now().strftime('%H:%M')}", flush=True)

      self.run_episode(i_episode)

      # plot graphs to the screen
      if self.log_level > 0: self.track.print_train_metrics(
          avg_only=(self.log_level == 1))
      if self.plot: self.track.plot()

      # check if we need to do any episode level updates (eg target network)
      self.agent.update_episode(i_episode)

      # test the target network and then save it
      if i_episode % self.params.test_freq == 0 and i_episode != 0:
        self.test() # this function should save the network

      # or only save the network
      elif i_episode % self.params.save_freq == 0:
        self.save()

    # the agent may require final updating at the end
    self.agent.update_episode(i_episode, finished=True)

    # save, log and plot now we are finished
    if self.log_level > 0:
      print("\nTraining complete, finished", i_episode, "episodes\n")

    # wrap up
    if self.render: self.env.render()
    if self.plot: self.track.plot()

  # need to define these functions for Trainer base class

  def test(self):
    """
    Empty test function, should be overriden for each environment.

    Outline:

    self.env.start_test() # if env needs to do extra logging
    self.agent.testing_mode() # if using pytorch, don't forget .eval()

    for i_episode in range(num_trials):
      self.run_episode(i_episode, test=True)

    test_data = self.env.get_test_data()

    self.env.end_test() # disable any test specific behaviour
    self.agent.training_mode() # disable any test time changes

    test_data = self.process_test_data(test_data) # maybe post-process
    self.save_test(test_data) # maybe save the results

    # return the test data
    return test_data
    """
    pass

if __name__ == "__main__":

  # put code here for testing and developing
  pass