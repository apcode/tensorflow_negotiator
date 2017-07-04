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

from negotiator import InputFn

INPUT_FILE="./data/val.txt"
RECORD_FILE="./data/val.txt.tfrecords"

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
            cell, 20, reuse=reuse)
        decoder = seq2seq.BasicDecoder(
            cell=out_cell, helper=helper,
            initial_state=out_cell.zero_state(
                dtype=tf.float32, batch_size=2))
        outputs = seq2seq.dynamic_decode(
            decoder=decoder, output_time_major=False,
            impute_finished=True, maximum_iterations=10)
        return outputs[0]


def TestBatchDecode():
    examples = tf.constant(np.random.randint(0, 5, size=(2, 10, 5)), dtype=np.float32)
    print(examples.shape)
    seq_length = tf.constant([7, 8], dtype=np.float32)
    print(seq_length.shape)
    outputs = tf.constant(np.random.randint(0, 5, size=(2, 1, 10)), dtype=np.float32)
    print(outputs.shape)
    training_helper = seq2seq.TrainingHelper(
        outputs, seq_length)
    train_outputs = decode(training_helper, "decode")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(train_outputs)
    

if __name__ == '__main__':
    #TestReadingFormat()
    #TestOutputVocab()
    #TestReadingTFRecords()
    TestBatchDecode()
