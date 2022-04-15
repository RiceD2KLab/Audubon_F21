#!/bin/ bash
DATA_DIR=""
IMG_EXT=""
DIR_IGNORE=""
NUM_WORKERS=""
MAX_ITERS=""
BATCH_SIZE=""
OUTPUT_DIR=""

echo "Welcome to Houston Audobon's state of the art bird detection model, developed in collaboration with Rice University"

echo "Please enter the path to the data directory (default: ./data):"
read DATA_DIR
if [[ -z "$DATA_DIR" ]]; then
  echo "DATA_DIR is empty. Going with default."
  DATA_DIR="./data"
fi
echo $DATA_DIR

echo "Please enter the image extension for the data (default: .JPEG):"
read IMG_EXT
if [[ -z "$IMG_EXT" ]]; then
  echo "IMG_EXT is empty. Going with default."
  IMG_EXT=".JPEG"
fi
echo $IMG_EXT

echo "Please enter the directories you wat to ignore in $DATA_DIR (default: ):"
read DIR_IGNORE
if [[ -z "$DIR_IGNORE" ]]; then
  echo "DIR_IGNORE is empty. Going with default."
  # TODO: NOT SURE HOW TO DEAL WITH LIST INPUTS
  DIR_IGNORE=[]
fi
echo $DIR_IGNORE

echo "Please enter the number of workers to use (default: 4):"
read NUM_WORKERS
if [[ -z "$NUM_WORKERS" ]]; then
  echo "NUM_WORKERS is empty. Going with default."
  NUM_WORKERS="4"
fi
echo $NUM_WORKERS

echo "Please enter the maximum number of iterations (default: 3000):"
read MAX_ITERS
if [[ -z "$MAX_ITERS" ]]; then
  echo "MAX_ITERS is empty. Going with default."
  MAX_ITERS="3000"
fi
echo $MAX_ITERS

echo "Please enter the batch size (default: 8):"
read BATCH_SIZE
if [[ -z "$BATCH_SIZE" ]]; then
  echo "BATCH_SIZE is empty. Going with default."
  BATCH_SIZE="8"
fi
echo $BATCH_SIZE

echo "Please enter the output directory (default: ./output):"
read OUTPUT_DIR
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR is empty. Going with default."
  OUTPUT_DIR="./output"
fi
echo $OUTPUT_DIR

echo "Going to train model:"

python Audubon_S22.py --data_dir=$DATA_DIR --img_ext=$IMG_EXT --dir_exceptions=$DIR_IGNORE --num_workers=$NUM_WORKERS
--max_iter=$MAX_ITERS --batch_size=$BATCH_SIZE --output_dir=$OUTPUT_DIR