"""SessionRunHooks for monitoring training.

PredictionSampleHook - prints predicted and target conversation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow import gfile

class PredictionSampleHook(tf.train.SessionRunHook):

  def __init__(self, output_dir, every_n_secs=None, every_n_steps=None):
    super(TrainingSampleHook, self).__init__()
    self._output_dir = os.path.join(output_dir, "samples")
    self._timer = SecondOrStepTimer(
        every_secs=every_n_secs,
        every_steps=every_n_steps)
    self._should_trigger = False
    self._iter_count = 0
    self._global_step = None

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()
    self._pred_dict = graph_utils.get_dict_from_collection("predictions")
    # Create the sample directory
    if self._sample_dir is not None:
      gfile.MakeDirs(self._sample_dir)

  def before_run(self, _run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      fetches = {
          "predicted_tokens": self._pred_dict["predicted_tokens"],
          "target_words": self._pred_dict["labels.target_tokens"],
          "target_len": self._pred_dict["labels.target_len"]
      }
      return tf.train.SessionRunArgs([fetches, self._global_step])
    return tf.train.SessionRunArgs([{}, self._global_step])

  def after_run(self, _run_context, run_values):
    result_dict, step = run_values.results
    self._iter_count = step

    if not self._should_trigger:
      return None

    # Convert dict of lists to list of dicts
    result_dicts = [
        dict(zip(result_dict, t)) for t in zip(*result_dict.values())
    ]

    # Print results
    result_str = ""
    result_str += "Prediction followed by Target @ Step {}\n".format(step)
    result_str += ("=" * 100) + "\n"
    for result in result_dicts:
      target_len = result["target_len"]
      predicted_slice = result["predicted_tokens"][:target_len - 1]
      target_slice = result["target_words"][1:target_len]
      result_str += self._target_delimiter.encode("utf-8").join(
          predicted_slice).decode("utf-8") + "\n"
      result_str += self._target_delimiter.encode("utf-8").join(
          target_slice).decode("utf-8") + "\n\n"
    result_str += ("=" * 100) + "\n\n"
    tf.logging.info(result_str)
    if self._sample_dir:
      filepath = os.path.join(self._sample_dir,
                              "samples_{:06d}.txt".format(step))
      with gfile.GFile(filepath, "w") as file:
        file.write(result_str)
    self._timer.update_last_triggered_step(self._iter_count - 1)
