"""SessionRunHooks for monitoring training.

TrainingSampleHook - prints outputs and target words.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow import gfile

class TrainingSampleHook(tf.train.SessionRunHook):
    """Sample the targets and labels during training.
    Converts word ids to text.
    """
    def __init__(self, outputs, targets, vocab,
                 every_secs=None, every_steps=None):
        super(TrainingSampleHook, self).__init__()
        self.outputs = outputs
        self.targets = targets
        self.vocab = vocab
        self.timer = SecondOrStepTimer(every_secs=every_secs,
                                       every_steps=every_steps)
        self.go = False
        self.step = 0
        self.global_step = None

    def begin(self):
        self.step = 0
        self.global_step = tf.train.get_global_step()

    def before_run(self, run_context):
        self.go = self.timer.should_trigger_for_step(self.step)
        if self.go:
            tensors = {
                "outputs": self.outputs,
                "targets": self.targets,
            }
            return tf.train.SessionRunArgs([tensors, self.global_step])
        return tf.train.SessionRunArgs([{}, self.global_step])

    def after_run(self, _, run_values):
        result_dict, step = run_values.results
        self.step = step
        if not self.go:
            return None
        # Just get first conversation in batch
        outputs = " ".join([self.vocab[wid]
                            for wid in result_dict["outputs"][0]])
        targets = " ".join([self.vocab[wid]
                            for wid in result_dict["targets"][0]])
        result = "TRAINING SAMPLE:\nTARGET(%d): %s\nOUTPUT(%d): %s" % (
            len(targets), targets, len(outputs), outputs)
        tf.logging.info(result)
        self.timer.update_last_triggered_step(self.step)
