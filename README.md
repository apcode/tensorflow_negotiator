# End-to-end Negotiation Network in Tensorflow

This based on Facebook's paper [Deal or No Deal? End-to-End Learning
for Negotiation Dialogues](https://arxiv.org/abs/1706.05125).  I have
implemented the basic process for training and interactive chat using
the Tensorflow contrib.seq2seq Decoder framework.

<< Still WIP >>

## Implemented
- pretraining on predicting words
- pretraining including input context vector to each word

## Still to do
- predict outputs using attention layer over complete conversation
- create loss combining word loss and output reward loss
- rl training on combined loss
- encode opponent words, generate your own
- negotiation chatbot

# Usage

The following are examples of how to use the applications. Get full help with
`--help` option on any of the programs.

The original dataset was provided by Facebook github [End-to-End Negotiator](https://github.com/facebookresearch/end-to-end-negotiator/tree/master/src/data/negotiate).
To transform input data into tensorflow Example format, an example:

    python process_input.py --input=data/train.txt --model_dir=data

To pretrain the model on predicting words only:

    python negotiator.py \
      --pretrain \
      --train_records=data/train.txt.tfrecords \
      --vocab_file=data/train.txt.vocab \
      --output_vocab=data/train.txt.outputs \
      --embedding_dimension=100 \
      --num_oov_vocab_buckets=10 \
      --model_dir=data/model \
      --learning_rate=0.01 \
      --batch_size=128 \
      --train_steps=1000 \

To train the model on goals:

    python negotiator.py \
      --train_goals \
      --train_records=data/train.txt.tfrecords \
      --vocab_file=data/train.txt.vocab \
      --output_vocab=data/train.txt.outputs \
      --model_dir=data/model \
      --learning_rate=0.01 \
      --batch_size=128 \
      --train_steps=1000

To chat:

    python negotiator.py \
      --chat \
      --vocab_file=data/train.txt.vocab \
      --output_vocab=data/train.txt.outputs \
      --model_dir=data/model

