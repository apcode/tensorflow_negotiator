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
    """Create two negotiations for each person."""
    input = re.search(r"<input>([^<]+)</input>", line)
    input = [int(x) for x in input.groups()[0].strip().split()]
    partner = re.search(r"<partner_input>([^<]+)</partner_input>", line)
    partner = [int(x) for x in partner.groups()[0].strip().split()]
    dialogue = re.search(r"<dialogue>(.+)</dialogue>", line)
    dialogue = dialogue.groups()[0].strip()
    dialogue = dialogue.replace("<eos> ", "")
    dialogue1 = dialogue.replace("THEM:", "<them>")
    dialogue1 = dialogue1.replace("YOU:", "<you>")
    dialogue2 = dialogue.replace("THEM:", "<you>")
    dialogue2 = dialogue2.replace("YOU:", "<them>")
    dialogue1 = dialogue1.split()
    dialogue2 = dialogue2.split()
    outputs = re.search(r"<output>(.+)</output>", line)
    outputs = outputs.groups()[0].strip().split()
    for i in range(len(outputs)):
        if "<disagree>" in outputs[i] \
           or "<no_agreement>" in outputs[i] \
           or "<disconnect>" in outputs[i]:
            outputs[i] = 0
        else:
            outputs[i] = int(outputs[i].split("=")[-1])
    outputs2 = outputs[3:]
    outputs = outputs[:3]
    return [{
        "input": input,
        "dialogue": dialogue1,
        "output": outputs,
    }, {
        "input": partner,
        "dialogue": dialogue2,
        "output": outputs2,
    }]


def ToExample(features):
    example = tf.train.Example()
    example.features.feature["input"].int64_list.value.extend(features["input"])
    example.features.feature["dialogue"].bytes_list.value.extend(features["dialogue"])
    example.features.feature["output"].int64_list.value.extend(features["output"])
    return example


def ProcessInputFile(infile, outfile, vocabfile):
    vocab = Counter()
    with tf.python_io.TFRecordWriter(outfile) as writer:
        for line in open(infile):
            inputs = ParseInput(line)
            for inp in inputs:
                vocab.update(inp["dialogue"])
                example = ToExample(inp)
                writer.write(example.SerializeToString())
    with open(vocabfile, "w") as f:
        # Write out vocab in most common first order
        for word in vocab.most_common():
            f.write(word[0] + '\n')


def main(_):
    base_input = os.path.basename(FLAGS.input)
    outfile = os.path.join(FLAGS.output_dir, base_input + ".tfrecords")
    vocabfile = os.path.join(FLAGS.output_dir, base_input + ".vocab")
    ProcessInputFile(FLAGS.input, outfile, vocabfile)


if __name__ == '__main__':
    tf.app.run()
