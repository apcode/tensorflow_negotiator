"""ContextTrainingHelper.

Identical to TrainingHelper except we add a context vector to each input step.
This allows you to provide an input context vector to each time step.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.contrib.seq2seq.python.ops.helper import (
    _unstack_ta, _transpose_batch_time)

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


class ContextTrainingHelper(seq2seq.TrainingHelper):

  def __init__(self, inputs, context, sequence_length, time_major=False,
               name=None):
    """Initializer. 
    Setup input_tas to include context at each step.
    Everything else is implemented in TrainingHelper.

    Args:
      inputs: A (structure of) input tensors.
      context: A (structure of) context tensors to concat to inputs.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.
    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    with ops.name_scope(name, "ContextTrainingHelper",
                        [inputs, context, sequence_length]):
        if isinstance(inputs, sparse_tensor.SparseTensor):
            inputs.dense_shape.set_shape([3])
        context = tf.expand_dims(context, 1)
        shape = tf.shape(inputs)[1]
        shape = tf.expand_dims(shape, 0)
        shape = tf.concat([tf.constant([1]),
                           tf.concat([shape, tf.constant([1])], -1)], 0)
        context = tf.tile(context, multiples=shape)
        inputs = tf.concat([inputs, context], axis=-1)
        if not time_major:
            inputs = nest.map_structure(_transpose_batch_time, inputs)
        self._input_tas = nest.map_structure(_unstack_ta, inputs)
        self._sequence_length = ops.convert_to_tensor(
            sequence_length, name="sequence_length")
        if self._sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be a vector, but received shape: %s" %
                self._sequence_length.get_shape())
        self._zero_inputs = nest.map_structure(
            lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
        self._batch_size = array_ops.size(sequence_length)


class ContextGreedyEmbeddingHelper(seq2seq.GreedyEmbeddingHelper):

    def __init__(self, embedding, context, start_tokens, end_token):
        """Initializer.
        Overrides embedding fn to concat the context vector each time.
        
        Args:
          embedding: A callable that takes a vector tensor of `ids` (argmax ids),
              or the `params` argument for `embedding_lookup`. The returned tensor
              will be passed to the decoder input.
          context: [batch_size, dim] tensor of context to append to input embedding.
          start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
          end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
          ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
              scalar.
        """
        self._context = context
        super(ContextGreedyEmbeddingHelper, self).__init__(
            embedding, start_tokens, end_token)
        # overwrite the embedding function
        self._embedding_only_fn = self._embedding_fn
        self._embedding_fn = lambda ids: tf.concat(
            [self._embedding_only_fn, self._context], -1)
