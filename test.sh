#!/bin/bash

function file_exists() {
   if [ ! -f $1 ]; then
      echo -e "File ${1} doesn't exist\nQuiting"
      exit
   fi
}

DATASETS=(facades maps edges2shoes edges2handbags)
SAMPLES=5

if [ $# -lt 2 ]; then
   echo -e "Usage ./test.sh <DATASET_NAME> <TEST_IMAGE_PATH>"
   exit
else
   DATASET=${1}
   TEST_IMG_PATH=${2}
   if [ $# -eq 3 ]; then
      SAMPLES=${3}
   fi
fi

file_exists ${TEST_IMG_PATH}

if [ ${DATASET} != ${DATASETS[2]} ]; then
   echo "The model is trained only on edges2shoes dataset"
   exit
fi

if [ ! -d test_samples ]; then
   mkdir test_samples
fi

python main.py --test \
--model bicycle \
--direction a2b \
--dataset ${DATASET} \
--full_summaries \
--batch_size 1 \
--sample_num ${SAMPLES} \
--ckpt ckpt/model_40.ckpt \
--test_source ${TEST_IMG_PATH}
