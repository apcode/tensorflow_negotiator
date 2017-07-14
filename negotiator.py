"""Learn to negotiate through conversations.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
import model


tf.flags.DEFINE_boolean("pretrain", False, "Pretrain the word model on word prediction only")
tf.flags.DEFINE_boolean("train_goals", False, "Train the word model on RL goals of negotiation")
tf.flags.DEFINE_boolean("chat", False, "Interactive chat with the model")
tf.flags.DEFINE_string("train_records", None, "TFRecords of training negotiations")
tf.flags.DEFINE_string("eval_records", None, "TFRecords to evaluate negotiations")
tf.flags.DEFINE_string("vocab_file", None, "Vocab file for word model")
tf.flags.DEFINE_integer("vocab_size", None, "Number of words in vocab")
tf.flags.DEFINE_string("output_vocab", None, "File containing tokens for output")
tf.flags.DEFINE_integer("embedding_dimension", 10, "Dimension of word embedding")
tf.flags.DEFINE_integer("num_oov_vocab_buckets", 20,
                        "Number of hash buckets to use for OOV words")
tf.flags.DEFINE_integer("num_items", 5, "Number of items to negotiate over")
tf.flags.DEFINE_string("model_dir", None, "Output directory to store model and summaries")

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for training")
tf.flags.DEFINE_float("clip_gradient", 5.0, "Clip gradient norm to this ratio")
tf.flags.DEFINE_integer("batch_size", 128, "Training minibatch size")
tf.flags.DEFINE_integer("train_steps", 1000,
                        "Number of train steps, None for continuous")
tf.flags.DEFINE_integer("eval_steps", 100, "Number of eval steps")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training data epochs")
tf.flags.DEFINE_integer("checkpoint_steps", 1000,
                        "Steps between saving checkpoints")
tf.flags.DEFINE_integer("num_threads", 1, "Number of reader threads")
tf.flags.DEFINE_boolean("debug", False, "Log debug info")
FLAGS = tf.flags.FLAGS

DEFAULT_CHAR=5  # Hardcoded to "." right now.
SEQUENCE_BUCKETS=[30, 50, 75, 100, 200]

def InputFn(input_file,
            batch_size,
            output,
            vocab_file,
            num_oov_vocab_buckets,
            embedding_dimension,
            sequence_bucketing_boundaries=None,
            num_epochs=None,
            num_threads=1):
    if num_epochs <= 0:
        num_epochs = None
    queue_capacity = max(100, batch_size * 4)
    vocab_size = len(open(vocab_file).readlines())
    def input_fn():
        with tf.name_scope('input'):
            file_queue = tf.train.string_input_producer([input_file])
            reader = tf.TFRecordReader()
            _, example = reader.read(file_queue)
            parse_spec = {
                "input": tf.FixedLenFeature([6], dtype=tf.int64),
                "dialogue": tf.VarLenFeature(dtype=tf.string),
                "output": tf.FixedLenFeature([3], dtype=tf.int64)
            }
            features = tf.parse_single_example(example, parse_spec)
            sequence_length = tf.shape(features["dialogue"])[0]
            features['sequence_length'] = sequence_length - 1
            word_lookup_table = tf.contrib.lookup.index_table_from_file(
                vocab_file, num_oov_vocab_buckets, vocab_size)
            features["dialogue"] = word_lookup_table.lookup(features["dialogue"])
            if sequence_bucketing_boundaries:
                _, batch_features = tf.contrib.training.bucket_by_sequence_length(
                    input_length=sequence_length,
                    tensors=features,
                    bucket_boundaries=sequence_bucket_boundaries,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    capacity=queue_capacity,
                    dynamic_pad=True)
            else:
                batch_features = tf.train.batch(
                    tensors=features,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    capacity=queue_capacity,
                    enqueue_many=False,
                    dynamic_pad=True)
            batch_features["dialogue"] = tf.sparse_tensor_to_dense(
                batch_features["dialogue"], default_value=DEFAULT_CHAR)
            batch_features["dialogue_next"] = batch_features["dialogue"][:, 1:]
            batch_features["dialogue"] = batch_features["dialogue"][:, :-1]
            word_embeddings = layers.embed_sequence(
                batch_features["dialogue"], vocab_size=vocab_size,
                embed_dim=embedding_dimension, scope="embedding")
            batch_features["embedded_dialogue"] = word_embeddings
        labels = batch_features.pop(output)
        return batch_features, labels
    return input_fn            


def Train(output_dir):
    params = {
        "num_units": 12,
        "vocab_file": FLAGS.vocab_file,
        "vocab_size": FLAGS.vocab_size,
        "batch_size": FLAGS.batch_size,
        "output_max_length": 66,
        "learning_rate": FLAGS.learning_rate,
    }
    print(params)
    estimator = model.Negotiator(
        pretrain=True,
        output_dir=output_dir,
        config=None,
        params=params)
    train_input = InputFn(FLAGS.train_records, FLAGS.batch_size, "dialogue_next",
                          FLAGS.vocab_file, FLAGS.num_oov_vocab_buckets,
                          FLAGS.embedding_dimension,
                          sequence_bucketing_boundaries=SEQUENCE_BUCKETS)
    print("STARTING TRAIN")
    estimator.train(train_input, steps=FLAGS.train_steps, hooks=None)
    print("TRAIN COMPLETE")


def main(_):
    if not FLAGS.vocab_size:
        FLAGS.vocab_size = len(open(FLAGS.vocab_file).readlines())
    if FLAGS.pretrain:
        learn_runner.run(experiment_fn=ExperimentFn(pretrain=True),
                         output_dir=FLAGS.model_dir)
    else:
        Train(FLAGS.model_dir)


if __name__ == '__main__':
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
