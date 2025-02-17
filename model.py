"""Negotiator model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.contrib.seq2seq import (
    BasicDecoder,
    dynamic_decode,
    GreedyEmbeddingHelper,
    TrainingHelper)
from helper import ContextTrainingHelper
from hooks import TrainingSampleHook

class Negotiator(tf.estimator.Estimator):

    def __init__(self, pretrain, output_dir, params, config):
        """ Initialize the Estimator """
        if pretrain:
            model_fn = self._pretrain_model_fn
        if "vocab_file" in params:
            self.vocab = [w.strip() for w in open(params["vocab_file"])]
        super(Negotiator, self).__init__(
            model_fn=model_fn,
            model_dir=output_dir,
            config=config,
            params=params)

    def _decode(self, helper, scope, reuse=None):
        """Build the decoder graph using seq2seq.BasicDecoder and a Helper.
        Args:
          helper: one of the seq2seq Helper classes used to provide the next input
                  when decoding.
        Returns:
          A Tensor of the outputs of the entire sequence decoding.
        """
        with tf.variable_scope(scope, reuse=reuse):
            cell = tf.contrib.rnn.LSTMCell(num_units=self.params["num_units"])
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                cell, self.params["vocab_size"], reuse=reuse)
            decoder = seq2seq.BasicDecoder(
                cell=out_cell, helper=helper,
                initial_state=out_cell.zero_state(
                    dtype=tf.float32, batch_size=self.params["batch_size"]))
            outputs, _, _ = seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=False,
                maximum_iterations=None)
            return outputs


    def _pretrain_model_fn(self, features, labels, mode, params=None, config=None):
        """Model fn for the estimator class to train purely on word probabilities.

        Follows requirements for tf.estimator.Estimators.
        - get embeddings used for input word ids. Used to generate embeddings
          as next input, when not training, or encoding.
        - provides training op and a prediction op over entire sequences.
        """
        transformed_input = tf.to_float(features["input"]) / tf.constant(10.0)
        training_helper = ContextTrainingHelper(
            inputs=features["embedded_dialogue"],
            context=transformed_input,
            sequence_length=features["sequence_length"],
            time_major=False)
        train_outputs = self._decode(training_helper, "decode")
        weights = tf.sequence_mask(features["sequence_length"], dtype=tf.float32)
        logits = train_outputs.rnn_output
        loss = seq2seq.sequence_loss(
            logits=logits,
            targets=labels,
            weights=weights)
        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer=params.get('optimizer', 'Adam'),
            learning_rate=params.get('learning_rate', 0.001),
            summaries=['loss', 'learning_rate'])
        words = tf.argmax(logits, axis=-1)
        run_hooks = [TrainingSampleHook(words, labels, self.vocab, every_steps=1000)]
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=None,  #pred_outputs.sample_id,
            loss=loss,
            train_op=train_op,
            training_hooks=run_hooks)
