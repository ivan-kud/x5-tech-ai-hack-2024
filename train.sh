#!/bin/bash

# export env variables
set -a
source .env
set +a

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3

# python ./model_train.py --model_name MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --cv_splits 10
python ./model_train.py --model_name MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
# python ./model_train.py --model_name sileod/deberta-v3-large-tasksource-nli --cv_splits 10
# python ./model_train.py --model_name sileod/deberta-v3-large-tasksource-nli
# python ./model_train.py --model_name microsoft/deberta-v2-xlarge-mnli --cv_splits 10
# python ./model_train.py --model_name microsoft/deberta-v2-xlarge-mnli
