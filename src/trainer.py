import os
import torch
import numpy as np
from dataclasses import dataclass, asdict
from itertools import count
import time
import random
from datetime import datetime
import functools

from modelsaver import ModelSaver

class TrackTraining:

  def __init__(self, test_metrics=None, avg_num=50, plt_frequency_seconds=30):
    """
    Track training data and generate matplotlib plots. To save test metrics pass
    in a list of metric names eg ["success_rate", "average_force", ...]. Metrics
    are stored as floats
    """
    self.numpy_float = np.float32
    self.avg_num = avg_num
    self.plt_frequency_seconds = plt_frequency_seconds
    # plotting options
    self.plot_episode_time = False
    self.plot_train_raw = False
    self.plot_train_avg = True
    self.plot_test_raw = True
    self.plot_test_metrics = False
    # general
    self.episodes_done = 0
    self.last_plot = 0
    self.per_action_time_taken = np.array([], dtype=self.numpy_float)
    self.avg_time_taken = np.array([], dtype=self.numpy_float)
    # training data
    self.train_episodes = np.array([], dtype=np.int32)
    self.train_rewards = np.array([], dtype=self.numpy_float)
    self.train_durations = np.array([], dtype=np.int32)
    self.train_avg_episodes = np.array([], dtype=np.int32)
    self.train_avg_rewards = np.array([], dtype=self.numpy_float)
    self.train_avg_durations = np.array([], dtype=np.int32)
    self.train_curriculum_stages = np.array([], dtype=np.int32)
    # testing data
    self.test_episodes = np.array([], dtype=np.int32)
    self.test_rewards = np.array([], dtype=self.numpy_float)
    self.test_durations = np.array([], dtype=np.int32)
    self.test_curriculum_stages = np.array([], dtype=np.int32)
    self.n_test_metrics = 0
    self.test_metric_names = []
    self.test_metric_values = []
    if test_metrics is not None: self.add_test_metrics(test_metrics)
    # misc
    self.fig = None
    self.axs = None

  def add_test_metrics(self, metrics_to_add, dtype=None, values=None):
    """
    Include additional test metrics
    """

    if metrics_to_add is None: return

    if dtype is None: dtype = self.numpy_float

    if isinstance(metrics_to_add, str):
      metrics_to_add = [metrics_to_add]
    if isinstance(values, np.ndarray):
      values = [values]

    for i, m in enumerate(metrics_to_add):
      self.test_metric_names.append(m)
      if values is None:
        thisvalue = np.array([], dtype=dtype)
      else:
        thisvalue = values[i]
      self.test_metric_values.append(thisvalue)

    self.n_test_metrics = len(self.test_metric_names)

  def get_test_metric(self, metric_name):
    """
    Return the array corresponding to a given metric_name
    """

    for i in range(len(self.test_metric_names)):
      if self.test_metric_names[i] == metric_name:
        return self.test_metrics[i]
    return None

  def log_training_episode(self, reward, duration, time_taken, curriculum_stage=0):
    """
    Log one training episode
    """

    self.train_episodes = np.append(self.train_episodes, self.episodes_done)
    self.train_durations = np.append(self.train_durations, duration)
    self.train_rewards = np.append(self.train_rewards, reward)
    self.train_curriculum_stages = np.append(self.train_curriculum_stages, curriculum_stage)
    self.per_action_time_taken = np.append(self.per_action_time_taken, time_taken)
    self.episodes_done += 1

    # update average information
    self.calc_static_average()

  def log_test_information(self, avg_reward, avg_duration, metrics=None):
    """
    Log information following a test
    """

    self.test_episodes = np.append(self.test_episodes, self.episodes_done)
    self.test_durations = np.append(self.test_durations, avg_duration)
    self.test_rewards = np.append(self.test_rewards, avg_reward)

    if metrics is not None:
      if len(metrics) != len(self.n_test_metrics):
        raise RuntimeError(f"TrackTraining.log_test_information got 'metrics' len={len(metrics)}, but self.n_test_metrics = {self.n_test_metrics}")
      for i in range(len(metrics)):
        self.test_metric_values[i] = np.append(self.test_metric_values[i], metrics[i])

  def calc_static_average(self):
    """
    Average rewards and durations to reduce data points
    """

    # find number of points we can average
    num_avg_points = len(self.train_avg_rewards) * self.avg_num

    # if we points which have not been averaged yet
    if num_avg_points + self.avg_num < len(self.train_episodes):

      # prepare to average rewards, durations, time taken
      unaveraged_r = self.train_rewards[num_avg_points:]
      unaveraged_d = self.train_durations[num_avg_points:]
      unaveraged_t = self.per_action_time_taken[num_avg_points:]

      num_points_to_avg = len(unaveraged_r) // self.avg_num

      for i in range(num_points_to_avg):
        # find average values
        avg_e = self.train_episodes[
          num_avg_points + (i * self.avg_num) + (self.avg_num // 2)]
        avg_r = np.mean(unaveraged_r[i * self.avg_num : (i + 1) * self.avg_num])
        avg_d = np.mean(unaveraged_d[i * self.avg_num : (i + 1) * self.avg_num])
        avg_t = np.mean(unaveraged_t[i * self.avg_num : (i + 1) * self.avg_num])
        # append to average lists
        self.train_avg_episodes = np.append(self.train_avg_episodes, avg_e)
        self.train_avg_rewards = np.append(self.train_avg_rewards, avg_r)
        self.train_avg_durations = np.append(self.train_avg_durations, avg_d)
        self.avg_time_taken = np.append(self.avg_time_taken, avg_t)

  def plot_matplotlib(self, xdata, ydata, ylabel, title, axs, label=None):
    """
    Plot a matplotlib 2x1 subplot
    """
    axs.plot(xdata, ydata, label=label)
    axs.set_title(title, fontstyle="italic")
    axs.set(ylabel=ylabel)

  def plot(self, plttitle=None, plt_frequency_seconds=None):
      """
      Plot training results figures, pass a frequency to plot only if enough
      time has elapsed
      """

      if plt_frequency_seconds is None:
        plt_frequency_seconds = self.plt_frequency_seconds

      # if not enough time has elapsed since the last plot
      if (self.last_plot + plt_frequency_seconds > time.time()):
        return

      self.plot_bar_chart = True

      if self.fig is None:
        # multiple figures
        self.fig = []
        self.axs = []
        if self.plot_train_raw: 
          fig1, axs1 = plt.subplots(2, 1)
          self.fig.append(fig1)
          self.axs.append(axs1)
        if self.plot_train_avg:
          fig2, axs2 = plt.subplots(2, 1)
          self.fig.append(fig2)
          self.axs.append(axs2)
        if self.plot_test_raw:
          fig3, axs3 = plt.subplots(2, 1)
          self.fig.append(fig3)
          self.axs.append(axs3)
        if self.plot_episode_time:
          fig4, axs4 = plt.subplots(1, 1)
          self.fig.append(fig4)
          self.axs.append([axs4, axs4]) # add paired to hold the pattern
        if self.plot_test_metrics:
          for i in range(self.n_test_metrics):
            fig5, axs5 = plt.subplots(1, 1)
            self.fig.append(fig5)
            self.axs.append([axs5, axs5]) # add paired to hold the pattern

      ind = 0

      E = "Episode"
      R = "Reward"
      D = "Duration"

      # clear all axes
      for i, pairs in enumerate(self.axs):
        if plttitle is not None:
          self.fig[i].suptitle(plttitle)
        for axis in pairs:
          axis.clear()

      if self.plot_train_raw:
        self.plot_matplotlib(self.train_episodes, self.train_durations, D,
                             "Raw durations", self.axs[ind][0])
        self.plot_matplotlib(self.train_episodes, self.train_rewards, R,
                             "Raw rewards", self.axs[ind][1])
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      if self.plot_train_avg:
        self.plot_matplotlib(self.train_avg_episodes, self.train_avg_durations, D,
                             f"Durations static average ({self.avg_num} samples)", 
                             self.axs[ind][0])
        self.plot_matplotlib(self.train_avg_episodes, self.train_avg_rewards, R,
                             f"Rewards static average ({self.avg_num} samples)", 
                             self.axs[ind][1])
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      if self.plot_test_raw:
        self.plot_matplotlib(self.test_episodes, self.test_durations, D,
                             "Test durations", self.axs[ind][0])
        self.plot_matplotlib(self.test_episodes, self.test_rewards, R,
                             "Test rewards", self.axs[ind][1])
        self.fig[ind].subplots_adjust(hspace=0.4)
        ind += 1

      # create plots for static average of time taken per step
      if self.plot_episode_time:
        self.plot_matplotlib(self.avgS_episodes, self.avg_time_taken, "Time per action / s",
          f"Time per action static average ({self.avg_num} samples)", self.axs[ind][0])
        ind += 1 

      if self.plot_test_metrics:
        for m, metric in enumerate(self.test_metric_names):
          self.plot_matplotlib(self.test_episodes, self.test_metric_values[m], f"{metric}",
            f"Test metric: {metric}", self.axs[ind][0])
          ind += 1 
        
      plt.pause(0.001)

      # save that we plotted
      self.last_plot = time.time()

      return

  def print_training(self):
    """
    Print out some training metrics
    """
    if self.episodes_done % self.avg_num == 0:
      if len(self.train_avg_rewards) == 0: return
      else: print(f"Episode {self.episodes_done}, avg_reward = {self.train_avg_rewards[-1]}")

  def get_avg_return(self):
    """
    Return the average reward only if the value has updated
    """

    if self.episodes_done % self.avg_num == 0:
      if len(self.train_avg_rewards) == 0: return None
      else: return self.train_avg_rewards[-1]

class Trainer:

  @dataclass
  class Parameters:

    num_episodes: int = 1_000
    test_freq: int = 200
    save_freq: int = 200
    use_curriculum: bool = False

  def __init__(self, agent, env, rngseed=None, device="cpu", log_level=1, plot=False,
               render=False, group_name="default_%Y-%m-%d", run_name="default_run_%H-%M",
               save=True, savedir="models", episode_log_rate=10, strict_seed=False):
    """
    Class that trains agents in an environment
    """

    # prepare class variables
    self.track = TrackTraining()
    self.params = Trainer.Parameters()
    self.agent = agent
    self.env = env
    self.saved_trainer_params = False
    self.last_loaded_agent_id = None
    self.last_saved_agent_id = None
    self.episode_fcn = None

    # input class options
    self.rngseed = rngseed
    self.device = device
    self.log_level = log_level
    self.plot = plot
    self.render = render
    self.log_rate_for_episodes = episode_log_rate
    
    # set up saving
    self.train_param_savename = "Trainer_params"
    self.track_savename = "Tracking_info"
    self.setup_saving(run_name, group_name, savedir, enable_saving=save)

    # are we plotting
    if self.plot:
      global plt
      import matplotlib.pyplot as plt
      plt.ion()

    # seed the environment (skip if given None for agent and env)
    # training only reproducible if torch.manual_seed() set BEFORE agent network initialisation
    self.training_reproducible = strict_seed
    if agent is not None and env is not None: self.seed(strict=strict_seed)
    else:
      if strict_seed or rngseed is not None:
        raise RuntimeError("Trainer.__init__() error: agent and/or env is None, environment is not seeded by rngseed or strict_seed was set")
      elif self.log_level >= 2:
        print("Trainer.__init__() warning: agent and/or env is None and environment is NOT seeded")

    if self.log_level > 0:
      print("Trainer settings:")
      print(" -> Run name:", self.run_name)
      print(" -> Group name:", self.group_name)
      print(" -> Given seed:", rngseed)
      print(" -> Training reproducible:", self.training_reproducible)
      print(" -> Using device:", self.device)
      print(" -> Save enabled:", self.enable_saving)
      if self.enable_saving:
        print(" -> Save path:", self.modelsaver.path)

  def setup_saving(self, run_name="default_run_%H-%M", group_name="default_%Y-%m-%d",
                   savedir="models", enable_saving=None):
    """
    Provide saving information and enable saving of models during training. The
    save() function will not work without first running this function
    """

    if enable_saving is not None:
      self.enable_saving = enable_saving

    # check for default group and run names (use current time and date)
    if group_name.startswith("default_"):
      group_name = datetime.now().strftime(group_name[8:])
    if run_name.startswith("default_"):
      run_name = f"{datetime.now().strftime(run_name[8:])}"
      
    # save information and create modelsaver to manage saving/loading
    self.group_name = group_name
    self.run_name = run_name
    self.savedir = savedir

    if self.enable_saving:
      self.modelsaver = ModelSaver(self.savedir + "/" + self.group_name,
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
      os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # increases GPU usage by 24MiB, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility and ctrl+f "CUBLAS_WORKSPACE_CONFIG"
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
      raise RuntimeError("Trainer.load() error: agentonly=True and trackonly=True, incompatible arguments")

    # check if modelsaver is defined
    if not hasattr(self, "modelsaver"):
      if path_to_run_folder is not None:
        print(f"load not given a modelsaver, making one from path_to_group: {path_to_run_folder}")
        self.modelsaver = ModelSaver(path_to_run_folder, log_level=self.log_level)
      elif group_name is not None:
        # try to find the group from this folder
        pathhere = os.path.dirname(os.path.abspath(__file__))
        print(f"load not given modelsaver or path_to_group, assuming group is local at {pathhere + '/' + self.savedir}")
        self.modelsaver = ModelSaver(pathhere + "/" + self.savedir + "/" + self.group_name,
                                     log_level=self.log_level)
      else:
        raise RuntimeError("load not given a modelsaver and either of a) path_to_run_folder b) group_name (if group can be found locally)")
    
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
          self.curriculum_change(self.curriculum_dict["stage"]) # apply initial stage settings

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
        self.track.log_training_episode(cumulative_reward, t + 1, time_per_step,
                                        curriculum_stage=0 if not self.params.use_curriculum
                                        else self.curriculum_dict["stage"])
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
        print(f"Trainer.train() warning: num_episodes={num_episodes_abs} (ignored) and num_episodes_extra={num_episodes_extra} (used) were both set. Training endpoing set as {self.params.num_episodes}")

    if i_start >= self.params.num_episodes:
      raise RuntimeError(f"Trainer.train() error: training episode start = {i_start} is greater or equal to the target number of episodes = {self.params.num_episodes}")

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
      print(f"\nBegin training, target is {self.params.num_episodes} episodes\n", flush=True)

    # prepare the agent for training
    self.agent.set_device(self.device)
    self.agent.training_mode()
    
    # begin training episodes
    for i_episode in range(i_start + 1, self.params.num_episodes + 1):

      if self.log_level == 1 and (i_episode - 1) % self.log_rate_for_episodes == 0:
        print(f"Begin training episode {i_episode}", flush=True)
      elif self.log_level > 1:
        print(f"Begin training episode {i_episode} at {datetime.now().strftime('%H:%M')}" + str_to_add, flush=True)

      self.run_episode(i_episode)

      # plot graphs to the screen
      if self.plot: self.track.plot(plt_frequency_seconds=1)

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
    if self.plot:
      self.track.plot(force=True, end=True, hang=True) # leave plots on screen if we are plotting

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