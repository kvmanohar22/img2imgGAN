#!/bin/bash

function file_exists() {
   if [ ! -f $1 ]; then
      echo -e "File ${1} doesn't exist\nQuiting"
      exit
   fi
}

DATASETS=(facades maps edges2shoes edges2handbags)

if [ $# -lt 3 ]; then
   echo -e "Usage ./test.sh <DATASET_NAME> <TEST_IMAGE_PATH> <CHECKPOINT_PATH>"
   exit
else
   DATASET=${1}
   TEST_IMG_PATH=${2}
   CHECKPOINT_PATH=${3}
fi


file_exists ${TEST_IMG_PATH}
file_exists ${CHECKPOINT_PATH}

if [ ${DATASET} != ${DATASETS[2]} ]; then
   echo "The model is trained only on edges2shoes dataset"
   exit
fi

python main.py --test \
--model bicycle \
--direction a2b \
--dataset ${DATASET} \
--full_summaries \
--batch_size 1 \
--sample_num 5 \
--ckpt ${CHECKPOINT_PATH} \
