import collections
import concurrent.futures
import functools
import json
import os
import re
import time

import numpy as np

from repos.elements.elements import path, printing, timer, when, counter

class Logger:

  def __init__(self, step, outputs, multiplier=1):
    assert outputs, 'Provide a list of logger outputs.'
    self.step = step
    self.outputs = outputs
    self.multiplier = multiplier
    self._last_step = None
    self._last_time = None
    self._metrics = []

  @timer.section('logger_add')
  def add(self, mapping, prefix=None):
    mapping = dict(mapping)
    # print('logger add:', len(mapping))
    assert len(mapping) <= 1000, list(mapping.keys())
    for key in mapping.keys():
      assert len(key) <= 200, (len(key), key[:200] + '...')
    step = int(self.step) * self.multiplier
    for name, value in mapping.items():
      name = f'{prefix}/{name}' if prefix else name
      if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, str):
        value = str(value)
      if not isinstance(value, str):
        value = np.asarray(value)
        if len(value.shape) not in (0, 1, 2, 3, 4):
          raise ValueError(
              f"Shape {value.shape} for name '{name}' cannot be "
              "interpreted as scalar, vector, image, or video.")
      self._metrics.append((step, name, value))

  def scalar(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) == 0, value.shape
    self.add({name: value})

  def vector(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) == 1, value.shape
    self.add({name: value})

  def image(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) in (2, 3), value.shape
    self.add({name: value})

  def video(self, name, value):
    value = np.asarray(value)
    assert len(value.shape) == 4, value.shape
    self.add({name: value})

  def text(self, name, value):
    assert isinstance(value, str), (type(value), str(value)[:100])
    self.add({name: value})

  @timer.section('logger_write')
  def write(self):
    if not self._metrics:
      return
    for output in self.outputs:
      with timer.section(type(output).__name__):
        output(tuple(self._metrics))
    self._metrics.clear()

  def close(self):
    self.write()
    for output in self.outputs:
      if hasattr(output, 'wait'):
        try:
          output.wait()
        except Exception as e:
          print(f'Error waiting on output: {e}')

class ProjectLogger(Logger):

  def __init__(self, log_rate, log_to_wandb=True, log_to_terminal=True, 
               log_to_JSON=True, wandb_instance=None, json_logdir="./",
               json_filename="metrics.jsonl"):
    """
    Logger that has multiple output streams:
      - terminal
      - weights and biases (wandb) [requires wandb_instance]
      - JSON files [requires logdir]

    Output streams are logged to at a specified log rate.

    Use this logger as follows:

    # initialise
    logger = ProjectLogger(...)

    # log data after every episode/training step
    logger.scalar('foo', 42)
    logger.scalar('foo', 43)
    logger.scalar('foo', 44)
    logger.vector('vector', np.zeros(100))
    logger.image('image', np.zeros((800, 600, 3, np.uint8)))
    logger.video('video', np.zeros((100, 64, 64, 3, np.uint8)))
    logger.log_step()

    For more see: https://github.com/danijar/elements
    """

    self.log_rate = log_rate
    self.log_when = when.Every(log_rate)
    step = counter.Counter()
    self.episodes_done = int(step)
    self.log_to_wandb = log_to_wandb
    self.log_to_JSON = log_to_JSON
    self.log_to_terminaml = log_to_terminal

    outputs = []

    if log_to_terminal:
      outputs.append(TerminalOutput())
    
    if log_to_wandb:
      if wandb_instance == None:
        print("ProjectLogger.__init__() error: log_to_wandb=True, but no wandb instance given. wandb logging disabled")
      else:
        outputs.append(WandBOutput(wandb_instance))

    if log_to_JSON:
      outputs.append(JSONLOutput(json_logdir, json_filename))

    if len(outputs) == 0:
      print("ProjectLogger.__init__() error: log_to_wandb, log_to_JSON, log_to_terminal are ALL False, nothing will be logged")

    super().__init__(step, outputs)

  def log_step(self, print_string=None):
    """
    Triggers a write action at the specified log rate, and increments the count
    of episodes done. Optionally print a string as well.
    """
    if self.log_when(self.episodes_done): 
      self.write()
      print(print_string)
    self.step.increment()
    self.episodes_done = int(self.step)


class AsyncOutput:

  def __init__(self, callback, parallel=True):
    self._callback = callback
    self._parallel = parallel
    if parallel:
      name = type(self).__name__
      self._worker = concurrent.futures.ThreadPoolExecutor(
          1, f'logger_{name}_async')
      self._future = None

  def wait(self):
    if self._parallel and self._future:
      concurrent.futures.wait([self._future])

  def __call__(self, summaries):
    if self._parallel:
      self._future and self._future.result()
      self._future = self._worker.submit(self._callback, summaries)
    else:
      self._callback(summaries)


class TerminalOutput:

  def __init__(self, pattern=r'.*', name=None, limit=50):
    self._pattern = (pattern != r'.*') and re.compile(pattern)
    self._name = name
    self._limit = limit

  def __call__(self, summaries):
    step = max(s for s, _, _, in summaries)
    scalars = {
        k: float(v) for _, k, v in summaries
        if isinstance(v, np.ndarray) and len(v.shape) == 0}
    if self._pattern:
      scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
    else:
      truncated = 0
      if len(scalars) > self._limit:
        truncated = len(scalars) - self._limit
        scalars = dict(list(scalars.items())[:self._limit])
    formatted = {k: self._format_value(v) for k, v in scalars.items()}
    if self._name:
      header = f'{"-" * 20}[{self._name} Step {step:_}]{"-" * 20}'
    else:
      header = f'{"-" * 20}[Step {step:_}]{"-" * 20}'
    content = ''
    if self._pattern:
      content += f"Metrics filtered by: '{self._pattern.pattern}'"
    elif truncated:
      content += f'{truncated} metrics truncated, filter to see specific keys.'
    content += '\n'
    if formatted:
      content += ' / '.join(f'{k} {v}' for k, v in formatted.items())
    else:
      content += 'No metrics.'
    printing.print_(f'\n{header}\n{content}\n', flush=True)

  def _format_value(self, value):
    value = float(value)
    if value == 0:
      return '0'
    elif 0.01 < abs(value) < 10000:
      value = f'{value:.2f}'
      value = value.rstrip('0')
      value = value.rstrip('0')
      value = value.rstrip('.')
      return value
    else:
      value = f'{value:.1e}'
      value = value.replace('.0e', 'e')
      value = value.replace('+0', '')
      value = value.replace('+', '')
      value = value.replace('-0', '-')
    return value


class JSONLOutput(AsyncOutput):

  def __init__(
      self, logdir, filename='metrics.jsonl', pattern=r'.*',
      strings=False, parallel=True):
    super().__init__(self._write, parallel)
    self._pattern = re.compile(pattern)
    self._strings = strings
    self.path_created = False
    self.given_filename = filename
    self.given_logdir = logdir

  @timer.section('jsonl')
  def _write(self, summaries):
    if not self.path_created:
      logdir = path.Path(self.given_logdir)
      logdir.mkdir()
      self._filename = logdir / self.given_filename
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str) and self._strings:
        bystep[step][name] = value
      if isinstance(value, np.ndarray) and len(value.shape) == 0:
        bystep[step][name] = float(value)
    lines = ''.join([
        json.dumps({'step': step, **scalars}) + '\n'
        for step, scalars in bystep.items()])
    printing.print_(f'Writing metrics: {self._filename}')
    with self._filename.open('a') as f:
      f.write(lines)


class TensorBoardOutput(AsyncOutput):

  def __init__(
      self, logdir, fps=20, videos=True, maxsize=1e9, parallel=True):
    super().__init__(self._write, parallel)
    self._logdir = str(path.Path(logdir))
    if self._logdir.startswith('/gcs/'):
      self._logdir = self._logdir.replace('/gcs/', 'gs://')
    self._fps = fps
    self._writer = None
    self._maxsize = self._logdir.startswith('gs://') and maxsize
    self._videos = videos
    if self._maxsize:
      self._checker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      self._promise = None
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.set_visible_devices([], 'TPU')

  @timer.section('tensorboard_write')
  def _write(self, summaries):
    import tensorflow as tf
    reset = False
    if self._maxsize:
      result = self._promise and self._promise.result()
      # print('Current TensorBoard event file size:', result)
      reset = (self._promise and result >= self._maxsize)
      self._promise = self._checker.submit(self._check)
    if not self._writer or reset:
      print('Creating new TensorBoard event file writer.')
      self._writer = self._retry(functools.partial(
          tf.summary.create_file_writer,
          self._logdir, max_queue=int(1e9), flush_millis=int(1e9)))
    self._writer.set_as_default()
    for step, name, value in summaries:
      try:
        if isinstance(value, str):
          self._retry(tf.summary.text, name, value, step)
        elif len(value.shape) == 0:
          self._retry(tf.summary.scalar, name, value, step)
        elif len(value.shape) == 1:
          if len(value) > 1024:
            value = value.copy()
            np.random.shuffle(value)
            value = value[:1024]
          self._retry(tf.summary.histogram, name, value, step)
        elif len(value.shape) == 2:
          self._retry(tf.summary.image, name, value[None, ..., None], step)
        elif len(value.shape) == 3:
          self._retry(tf.summary.image, name, value[None], step)
        elif len(value.shape) == 4 and self._videos:
          self._video_summary(name, value, step)
      except Exception:
        print('Error writing summary:', name)
        raise
    self._writer.flush()

  @timer.section('tensorboard_check')
  def _check(self):
    import tensorflow as tf
    events = tf.io.gfile.glob(self._logdir.rstrip('/') + '/events.out.*')
    return tf.io.gfile.stat(sorted(events)[-1]).length if events else 0

  def _retry(self, fn, *args, attempts=3, delay=(3, 10)):
    import tensorflow as tf
    for retry in range(attempts):
      try:
        return fn(*args)
      except tf.errors.PermissionDeniedError as e:
        if retry >= attempts - 1:
          raise
        print(f'Retrying after exception: {e}')
        delay and time.sleep(float(np.random.uniform(*delay)))

  @timer.section('tensorboard_video')
  def _video_summary(self, name, video, step):
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    name = name if isinstance(name, str) else name.decode('utf-8')
    assert video.dtype in (np.float32, np.uint8), (video.shape, video.dtype)
    if np.issubdtype(video.dtype, np.floating):
      video = np.clip(255 * video, 0, 255).astype(np.uint8)
    try:
      T, H, W, C = video.shape
      summary = tf1.Summary()
      image = tf1.Summary.Image(height=H, width=W, colorspace=C)
      image.encoded_image_string = _encode_gif(video, self._fps)
      summary.value.add(tag=name, image=image)
      content = summary.SerializeToString()
      self._retry(tf.summary.experimental.write_raw_pb, content, step)
    except (IOError, OSError) as e:
      print('GIF summaries require ffmpeg in $PATH.', e)
      self._retry(tf.summary.image, name, video, step)


class WandBOutput:

  def __init__(self, wandb_instance=None, pattern=r'.*', **kwargs):
    self._pattern = re.compile(pattern)
    if wandb_instance is None:
      import wandb
      wandb.init(**kwargs)
      self._wandb = wandb
    else: self._wandb = wandb_instance

  def __call__(self, summaries):
    bystep = collections.defaultdict(dict)
    wandb = self._wandb
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str):
        bystep[step][name] = value
      elif len(value.shape) == 0:
        bystep[step][name] = float(value)
      elif len(value.shape) == 1:
        bystep[step][name] = wandb.Histogram(value)
      elif len(value.shape) in (2, 3):
        value = value[..., None] if len(value.shape) == 2 else value
        assert value.shape[3] in [1, 3, 4], value.shape
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        value = np.transpose(value, [2, 0, 1])
        bystep[step][name] = wandb.Image(value)
      elif len(value.shape) == 4:
        assert value.shape[3] in [1, 3, 4], value.shape
        value = np.transpose(value, [0, 3, 1, 2])
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        bystep[step][name] = wandb.Video(value)

    for step, metrics in bystep.items():
      self._wandb.log(metrics, step=step)


class ScopeOutput(AsyncOutput):

  def __init__(self, logdir, fps=10, pattern=r'.*'):
    super().__init__(self._write, parallel=True)
    import scope
    import scope.writer
    scope.writer.FPS = fps
    logdir = path.Path(logdir)
    self.writer = scope.Writer(logdir)
    self.pattern = (pattern != r'.*') and re.compile(pattern)

  def _write(self, summaries):
    for step, name, value in summaries:
      if self.pattern and not self.pattern.search(name):
        continue
      self.writer.add(step, {name: value})
    self.writer.flush()


class MLFlowOutput:

  def __init__(self, run_name=None, resume_id=None, config=None, prefix=None):
    import mlflow
    self._mlflow = mlflow
    self._prefix = prefix
    self._setup(run_name, resume_id, config)

  def __call__(self, summaries):
    bystep = collections.defaultdict(dict)
    for step, name, value in summaries:
      if len(value.shape) == 0 and self._pattern.search(name):
        name = f'{self._prefix}/{name}' if self._prefix else name
        bystep[step][name] = float(value)
    for step, metrics in bystep.items():
      self._mlflow.log_metrics(metrics, step=step)

  def _setup(self, run_name, resume_id, config):
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'local')
    run_name = run_name or os.environ.get('MLFLOW_RUN_NAME')
    resume_id = resume_id or os.environ.get('MLFLOW_RESUME_ID')
    print('MLFlow Tracking URI:', tracking_uri)
    print('MLFlow Run Name:    ', run_name)
    print('MLFlow Resume ID:   ', resume_id)
    if resume_id:
      runs = self._mlflow.search_runs(None, f'tags.resume_id="{resume_id}"')
      assert len(runs), ('No runs to resume found.', resume_id)
      self._mlflow.start_run(run_name=run_name, run_id=runs['run_id'].iloc[0])
      for key, value in config.items():
        self._mlflow.log_param(key, value)
    else:
      tags = {'resume_id': resume_id or ''}
      self._mlflow.start_run(run_name=run_name, tags=tags)


@timer.section('gif')
def _encode_gif(frames, fps):
  from subprocess import Popen, PIPE
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tobytes())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out
