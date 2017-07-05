"""Convert Facebook's negotiation input to TFRecords:
Format:
  input: values of items
  dialogue: full dialog including special tokens
  output: ??
  partner_input: ??
"""
import os.path
import re
import tensorflow as tf
from collections import Counter

tf.flags.DEFINE_string("input", None, "Facebook format negotiation input")
tf.flags.DEFINE_string("output_dir", ".", "Directory to store output records")
FLAGS = tf.flags.FLAGS


def ParseInput(line):
    input = re.search(r"<input>([^<]+)</input>", line)
    input = [int(x) for x in input.groups()[0].strip().split()]
    dialogue = re.search(r"<dialogue>(.+)</dialogue>", line)
    dialogue = dialogue.groups()[0].strip()
    dialogue = dialogue.replace("THEM:", "<them>")
    dialogue = dialogue.replace("YOU:", "<you>")
    dialogue = dialogue.replace("<eos> ", "")
    dialogue = dialogue.split()
    output = re.search(r"<output>(.+)</output>", line)
    output = output.groups()[0].strip().split()
    partner = re.search(r"<partner_input>([^<]+)</partner_input>", line)
    partner = [int(x) for x in partner.groups()[0].strip().split()]
    assert len(input) == len(output) and len(input) == len(partner)
    return {
        "input": input,
        "dialogue": dialogue,
        "output": output,
        "partner": partner
    }


def ToExample(features):
    example = tf.train.Example()
    example.features.feature["input"].int64_list.value.extend(features["input"])
    example.features.feature["dialogue"].bytes_list.value.extend(features["dialogue"])
    example.features.feature["output"].bytes_list.value.extend(features["output"])
    example.features.feature["partner_input"].int64_list.value.extend(features["partner"])
    return example


def ProcessInputFile(infile, outfile, vocabfile, outtokfile):
    vocab = Counter()
    outputs = Counter()
    with tf.python_io.TFRecordWriter(outfile) as writer:
        for line in open(infile):
            inputs = ParseInput(line)
            vocab.update(inputs["dialogue"])
            outputs.update(inputs["output"])
            example = ToExample(inputs)
            writer.write(example.SerializeToString())
    with open(vocabfile, "w") as f:
        # Write out vocab in most common first order
        for word in vocab.most_common():
            f.write(word[0] + '\n')
    with open(outtokfile, "w") as f:
        # Write out vocab in most common first order
        for tok in outputs.most_common():
            f.write(tok[0] + '\n')


def main(_):
    base_input = os.path.basename(FLAGS.input)
    outfile = os.path.join(FLAGS.output_dir, base_input + ".tfrecords")
    vocabfile = os.path.join(FLAGS.output_dir, base_input + ".vocab")
    outtokfile = os.path.join(FLAGS.output_dir, base_input + ".outputs")
    ProcessInputFile(FLAGS.input, outfile, vocabfile, outtokfile)


if __name__ == '__main__':
    tf.app.run()
