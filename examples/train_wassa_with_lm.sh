#!/usr/bin/env bash

GLUE_DIR=/data/datasets
BERT_PYTORCH_DIR=/data/checkpoints/wassalm_uncased_L-12_H-768_A-12

python run_classifier.py \
	--task_name wassa6 \
	--do_train \
	--do_eval \
	--do_predict \
	--bert_model $BERT_PYTORCH_DIR/ \
	--data_dir $GLUE_DIR/Wassa2018/ \
	--max_seq_length 256 \
	--train_batch_size 128 \
	--learning_rate 2e-5 \
	--num_train_epochs 1.0 \
	--output_dir /data/outputs/CHANGEME/ \
	$@
