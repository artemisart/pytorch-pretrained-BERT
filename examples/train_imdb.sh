#!/usr/bin/env bash

GLUE_DIR=/data/datasets
BERT_BASE_DIR=/data/checkpoints/uncased_L-12_H-768_A-12
BERT_PYTORCH_DIR=$BERT_BASE_DIR

python run_classifier.py \
	--task_name IMDB \
	--do_train \
	--do_eval \
	--do_lower_case \
	--data_dir $GLUE_DIR/IMDB/ \
	--vocab_file $BERT_BASE_DIR/vocab.txt \
	--bert_config_file $BERT_BASE_DIR/bert_config.json \
	--init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
	--max_seq_length 256 \
	--train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir /tmp/imdb_output/ \
	--optimize_on_cpu \
	$@
