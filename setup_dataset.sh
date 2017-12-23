#!/bin/bash

DATASET_ROOT="data"

# Function to check if a directory exists
function directory_exists() {
   if [ ! -d $1 ]; then
      echo -e "Directory \"$1\" doesn't exist!\nExiting"
      exit
   fi
}


# Function to generate the dataset file
function generate_dataset_file() {
   directory_exists $1
   echo -e "Generating dataset files for $1"
   cd $1
   ls train | sort > train.txt
   ls val | sort > val.txt
   if [ -d test ]; then
      ls test | sort > test.txt
   fi
   cd ..
}

# Generate dataset file
directory_exists ${DATASET_ROOT}
cd ${DATASET_ROOT}

DATASETS=(facades maps edges2shoes edges2handbags)
for file in ${DATASETS[@]}; do
   generate_dataset_file $file
done
echo -e "Done !"
