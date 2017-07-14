"""Test reading and processing negotiator input text.

Format from facebook's end-to-end-negotiator.

Pre line:
<input> item values </input>
<dialogue> THEM: text <eos> YOU: text <eos> ... THEM: <selection> </dialogue>
<output> item0=0 item1=0 item2=1 ... </output>
<partner_input> partner item values </partner_input>

TFRecord:
input: [int64]
dialogue: [string: <them> text <you> text <them> <selection>]
output: [int64]
partner_input: [int64]
"""
import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from helper import ContextTrainingHelper
from negotiator import InputFn

INPUT_FILE="./data/val.txt"
RECORD_FILE="./data/val.txt.tfrecords"
VOCAB_FILE="./data/val.txt.vocab"
VOCAB_SIZE=len(open(VOCAB_FILE).readlines())
BATCH_SIZE=4

def TestReadingFormat():
    with open(INPUT_FILE) as f:
        features = {}
        line = f.readline()
        input = re.search(r"<input>([^<]+)</input>", line)
        input = [int(x) for x in input.groups()[0].strip().split()]
        features["input"] = input
        dialogue = re.search(r"<dialogue>(.+)</dialogue>", line)
        dialogue = dialogue.groups()[0].strip()
        dialogue = dialogue.replace("THEM:", "<them>")
        dialogue = dialogue.replace("YOU:", "<you>")
        dialogue = dialogue.replace("<eos> ", "")
        features["dialogue"] = dialogue
        output = re.search(r"<output>([^<]+)</output>", line)
        output = [int(x) for x in re.findall(r'item\d=(\d)', output.groups()[0].strip())]
        features["output"] = output
        partner = re.search(r"<partner_input>([^<]+)</partner_input>", line)
        partner = [int(x) for x in partner.groups()[0].strip().split()]
        features["partner_input"] = partner
        print features


def TestOutputVocab():
    tokens = set()
    for line in open(INPUT_FILE):
        try:
            output = re.search(r"<output>(.+)</output>", line)
            output = output.groups()[0].strip()
            tokens.update(output.split())
        except:
            print "ERROR", line
    print tokens


def TestReadingTFRecords():
    parse_spec = {
        "input": tf.FixedLenFeature([6], dtype=tf.int64),
        "dialogue": tf.VarLenFeature(dtype=tf.string),
        "output": tf.FixedLenFeature([6], dtype=tf.string)
    }
    sess = tf.Session()
    reader = tf.python_io.tf_record_iterator(RECORD_FILE)
    record = reader.next()
    example = [tf.parse_single_example(record, parse_spec)]
    record = reader.next()
    example.append(tf.parse_single_example(record, parse_spec))
    print example


def TestInputFn():
    train_input = InputFn(RECORD_FILE, BATCH_SIZE, "dialogue_next",
                          VOCAB_FILE, 10, 10)
    features, labels = train_input()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        f, l = sess.run([features, labels])
        for k, v in f.items():
            print(k, v.shape)
        print(l.shape)
        print(f["dialogue"])
        print(l)
        print(f["sequence_length"])
        coord.request_stop()
        coord.join(threads)


def decode(helper, scope, reuse=None):
    """Build the decoder graph using seq2seq.BasicDecoder and a Helper.
    Args:
      helper: one of the seq2seq Helper classes used to provide the next input
              when decoding.
    Returns:
      A Tensor of the outputs of the entire sequence decoding.
    """
    with tf.variable_scope(scope, reuse=reuse):
        cell = tf.contrib.rnn.LSTMCell(num_units=5)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell, VOCAB_SIZE, reuse=reuse)
        decoder = seq2seq.BasicDecoder(
            cell=out_cell, helper=helper,
            initial_state=out_cell.zero_state(
                dtype=tf.float32, batch_size=BATCH_SIZE))
        outputs = seq2seq.dynamic_decode(
            decoder=decoder, output_time_major=False,
            impute_finished=True, maximum_iterations=None)
        return outputs[0]


def TestBatchDecode():
    train_input = InputFn(RECORD_FILE, BATCH_SIZE, "dialogue_next",
                          VOCAB_FILE, 1, 20)
    features, labels = train_input()
    training_helper = seq2seq.TrainingHelper(
        inputs=features["embedded_dialogue"],
        sequence_length=features["sequence_length"],
        time_major=False)
    train_outputs = decode(training_helper, "decode")
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        outputs = sess.run(train_outputs)
        print(outputs.rnn_output.shape)
        coord.request_stop()
        coord.join(threads)


def TestSequenceLoss():
    train_input = InputFn(RECORD_FILE, BATCH_SIZE, "dialogue_next",
                          VOCAB_FILE, 1, 20)
    features, labels = train_input()
    training_helper = seq2seq.TrainingHelper(
        inputs=features["embedded_dialogue"],
        sequence_length=features["sequence_length"],
        time_major=False)
    train_outputs = decode(training_helper, "decode")
    logits = train_outputs.rnn_output
    weights = tf.sequence_mask(features["sequence_length"], dtype=tf.float32)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        num_classes = tf.shape(logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        labels_flat = tf.reshape(labels, [-1])
        weights_flat = tf.reshape(weights, [-1])
        labels_flat = tf.to_float(labels_flat)
        labels_flat *= weights_flat
        weights, logits, labels = sess.run([weights_flat, logits_flat, labels_flat])
        print(logits)
        print(labels)
        print(weights)
        print(logits.shape)
        print(labels.shape)
        print(weights.shape)
        coord.request_stop()
        coord.join(threads)


def TestConcatContext():
    examples = tf.constant(np.random.randint(0, 5, size=(2, 10, 5)), dtype=np.float32)
    print(examples.shape)
    context = tf.constant(np.random.randint(0, 2, size=(5,)), dtype=np.float32)
    context = tf.expand_dims(tf.expand_dims(context, 0), 0)
    shp = tf.expand_dims(tf.shape(examples)[:-1], 0)
    shp = tf.squeeze(tf.concat([shp, tf.constant(1, shape=[1,1])], -1))
    print(context.shape)
    print(shp)
    context = tf.tile(context, multiples=shp)
    examples2 = tf.concat([examples, context], axis=-1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([examples, context]))
        ex2 = sess.run(examples2)
        print(ex2)
        print(ex2.shape)

if __name__ == '__main__':
    #TestReadingFormat()
    #TestOutputVocab()
    #TestReadingTFRecords()
    #TestInputFn()
    #TestBatchDecode()
    #TestSequenceLoss()
    TestConcatContext()
