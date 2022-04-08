#!/bin/ bash
DATA_DIR=""
IMG_EXT=""
DIR_IGNORE=""
MODEL_TYPE=""
MODEL_CONFIG_FILE=""
PRETRAINED_WEIGHTS_FILE=""
NUM_WORKERS=""
EVAL_PERIOD=""
MAX_ITERS=""
CHECKPOINT_PERIOD=""
LEARNING_RATE=""
SOLVER_WARMUP_FACTOR=""
SOLVER_WARMUP_ITERS=""
SCHEDULER_GAMMA=""
SCHEDULER_STEPS=""
WEIGHT_DECAY=""
BATCH_SIZE=""
FOCAL_LOSS_GAMMA=""
FOCAL_LOSS_ALPHA=""
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

echo "Please enter the model type. Options: retinanet or faster-rcnn (default: faster-rcnn):"
read MODEL_TYPE
if [[ -z "$MODEL_TYPE" ]]; then
  echo "MODEL_TYPE is empty. Going with default."
  MODEL_TYPE="faster-rcnn"
fi
echo $MODEL_TYPE

echo "Please enter the path to the model config file (default: COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml):"
read MODEL_CONFIG_FILE
if [[ -z "$MODEL_CONFIG_FILE" ]]; then
  echo "MODEL_CONFIG_FILE is empty. Going with default."
  MODEL_CONFIG_FILE="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
fi
echo $MODEL_CONFIG_FILE

echo "Please enter the path to the pretrained weights file (default: ):"
read PRETRAINED_WEIGHTS_FILE
if [[ -z "$PRETRAINED_WEIGHTS_FILE" ]]; then
  echo "PRETRAINED_WEIGHTS_FILE is empty. Going with default."
fi
echo $PRETRAINED_WEIGHTS_FILE

echo "Please enter the number of workers to use (default: 4):"
read NUM_WORKERS
if [[ -z "$NUM_WORKERS" ]]; then
  echo "NUM_WORKERS is empty. Going with default."
  NUM_WORKERS="4"
fi
echo $NUM_WORKERS

echo "Please enter the evaluation period (default: 0):"
read EVAL_PERIOD
if [[ -z "$EVAL_PERIOD" ]]; then
  echo "EVAL_PERIOD is empty. Going with default."
  EVAL_PERIOD="0"
fi
echo $EVAL_PERIOD

echo "Please enter the maximum number of iterations (default: 3000):"
read MAX_ITERS
if [[ -z "$MAX_ITERS" ]]; then
  echo "MAX_ITERS is empty. Going with default."
  MAX_ITERS="3000"
fi
echo $MAX_ITERS

echo "Please enter the checkpoint period (default: 1000):"
read CHECKPOINT_PERIOD
if [[ -z "$CHECKPOINT_PERIOD" ]]; then
  echo "CHECKPOINT_PERIOD is empty. Going with default."
  CHECKPOINT_PERIOD="1000"
fi
echo $CHECKPOINT_PERIOD

echo "Please enter the learning rate (default: 1e-3):"
read LEARNING_RATE
if [[ -z "$LEARNING_RATE" ]]; then
  echo "LEARNING_RATE is empty. Going with default."
  LEARNING_RATE="1e-3"
fi
echo $LEARNING_RATE

echo "Please enter the solver warmup factor (default: 0.001):"
read SOLVER_WARMUP_FACTOR
if [[ -z "$SOLVER_WARMUP_FACTOR" ]]; then
  echo "SOLVER_WARMUP_FACTOR is empty. Going with default."
  SOLVER_WARMUP_FACTOR="0.001"
fi
echo $SOLVER_WARMUP_FACTOR

echo "Please enter the number of solver warmup iterations (default: 100):"
read SOLVER_WARMUP_ITERS
if [[ -z "$SOLVER_WARMUP_ITERS" ]]; then
  echo "SOLVER_WARMUP_ITERS is empty. Going with default."
  SOLVER_WARMUP_ITERS="100"
fi
echo $SOLVER_WARMUP_ITERS

echo "Please enter the scheduler gamma value (default: 0.1):"
read SCHEDULER_GAMMA
if [[ -z "$SCHEDULER_GAMMA" ]]; then
  echo "SCHEDULER_GAMMA is empty. Going with default."
  SCHEDULER_GAMMA="0.1"
fi
echo $SCHEDULER_GAMMA

echo "Please enter the number of steps on scheduler (default: 1500):"
read SCHEDULER_STEPS
if [[ -z "$SCHEDULER_STEPS" ]]; then
  echo "SCHEDULER_STEPS is empty. Going with default."
  SCHEDULER_STEPS=[1500]
fi
echo $SCHEDULER_STEPS

echo "Please enter the weight decay (default: 1e-4):"
read WEIGHT_DECAY
if [[ -z "$WEIGHT_DECAY" ]]; then
  echo "WEIGHT_DECAY is empty. Going with default."
  WEIGHT_DECAY="1e-4"
fi
echo $WEIGHT_DECAY

echo "Please enter the batch size (default: 8):"
read BATCH_SIZE
if [[ -z "$BATCH_SIZE" ]]; then
  echo "BATCH_SIZE is empty. Going with default."
  BATCH_SIZE="8"
fi
echo $BATCH_SIZE

echo "Please enter the focal loss gamma (default:  2.0):"
read FOCAL_LOSS_GAMMA
if [[ -z "$FOCAL_LOSS_GAMMA" ]]; then
  echo "FOCAL_LOSS_GAMMA is empty. Going with default."
  FOCAL_LOSS_GAMMA="2.0"
fi
echo $FOCAL_LOSS_GAMMA

echo "Please enter the focal loss alpha (default: 0.25):"
read FOCAL_LOSS_ALPHA
if [[ -z "$FOCAL_LOSS_ALPHA" ]]; then
  echo "FOCAL_LOSS_ALPHA is empty. Going with default."
  FOCAL_LOSS_ALPHA="0.25"
fi
echo $FOCAL_LOSS_ALPHA

echo "Please enter the output directory (default: ./output):"
read OUTPUT_DIR
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR is empty. Going with default."
  OUTPUT_DIR="./output"
fi
echo $OUTPUT_DIR

echo "Going to train model:"

python Audubon_S22.py --data_dir=$DATA_DIR --img_ext=$IMG_EXT --dir_exceptions=$DIR_IGNORE --model_type=$MODEL_TYPE
--model_config_file=$MODEL_CONFIG_FILE --pretrained_weights_file=$PRETRAINED_WEIGHTS_FILE --num_workers=$NUM_WORKERS
--eval_period=$EVAL_PERIOD --max_iter=$MAX_ITERS --checkpoint_period=$CHECKPOINT_PERIOD --learning_rate=$LEARNING_RATE
--solver_warmup_factor=$SOLVER_WARMUP_FACTOR --solver_warmup_iters=$SOLVER_WARMUP_ITERS --scheduler_gamma=$SCHEDULER_GAMMA
--scheduler_steps=$SCHEDULER_STEPS --weight_decay=$WEIGHT_DECAY --batch_size=$BATCH_SIZE --focal_loss_gamma=$FOCAL_LOSS_GAMMA
--focal_loss_alpha=$FOCAL_LOSS_ALPHA --output_dir=$OUTPUT_DIR