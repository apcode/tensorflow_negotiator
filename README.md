# Negotiation Network in Tensorflow

This based on the paper [ref]().  I have implemented the main basic
process for training and interactive chat using the Tensorflow Decoder
OO framework.

<< Still WIP >>

# Usage

The following are examples of how to use the applications. Get full help with
`--help` option on any of the programs.

The original dataset was provided by Facebook (end-to-end-negotiator)
[https://github.com/facebookresearch/end-to-end-negotiator/tree/master/src/data/negotiate].
To transform input data into tensorflow Example format, an example:

    process_input.py --input=./data/negotiations.txt --model*dir=.

To pretrain the model on predicting words only:

    nogotiator.py \
      --pretrain \
      --train_records=./data/negotiations.tfrecords \
      --vocab_file=./data/vocab.txt \
      --vocab_size=N \
      --model_dir=model \

To train the model on goals:

    nogotiator.py \
      --train_goals \
      --train_records=./data/negotiations.tfrecords \
      --vocab_file=./data/vocab.txt \
      --vocab_size=N \
      --model_dir=model \

To chat:

    negotiator.py \
      --chat \
      --vocab_file=./data/vocab.txt \
      --vocab_size=N \
      --model_dir=model
