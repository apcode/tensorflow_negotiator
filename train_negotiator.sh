#!/bin/bash

set +v 

DATADIR=data
MODELDIR=$DATADIR/model
DATASET=$DATADIR/train.txt.tfrecords
VOCAB=$DATADIR/train.txt.vocab
OUTPUT_VOCAB=$DATADIR/train.txt.outputs

mkdir -p $MODELDIR

python negotiator.py \
    --train_records=$DATASET \
    --vocab_file=$VOCAB \
    --output_vocab=$OUTPUT_VOCAB \
    --embedding_dimension=100 \
    --num_oov_vocab_buckets=10 \
    --model_dir=$MODELDIR \
    --learning_rate=0.01 \
    --batch_size=128 \
    --train_steps=1000 \
    --checkpoint_steps=100 \
    --debug
