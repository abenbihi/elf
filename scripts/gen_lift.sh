#!/bin/sh

# WARNING: Run this script with python2 ... 
# or prepare to suffer ... 
# and I will laugh at your suffering

export PYTHONPATH=$PYTHONPATH:/home/ws/methods/lift/tf-lift

PYTHON=python3

if [ "$#" -eq 0 ]; then
  echo "1. lift_data_id (lift kp and des are stored in res/lift/id)"
  exit 0
fi

if [ "$#" -ne 1 ]; then
  echo "Error: bad number of arguments"
  exit 1
fi

lift_data_id="$1"

if [ 1 -eq 1 ]; then
  # rotation augmentation for v_ images
  LOG_DIR='meta/weights/lift/release-aug/' # path to trained weights
  AUG_OPTIONS='--use_batch_norm=False --mean_std_type=hardcoded'
  
  # kp: run OK
  "$PYTHON" -m methods.lift.evaluation.main --task=test \
    --subtask=kp \
    --logdir="$LOG_DIR" \
    --test_img_file=toto.png \
    --test_out_file=toto.txt \
    "$AUG_OPTIONS"
  
  if [ $? -ne 0 ]; then
    echo "Failed kp run"
    exit 1
  fi
  
  # ori: run OK
  "$PYTHON" -m methods.lift.evaluation.main \
    --task=test --subtask=ori \
    --logdir="$LOG_DIR" \
    --test_img_file=toto.png \
    --test_out_file=toto.txt \
    --test_kp_file=toto.txt \
    "$AUG_OPTIONS"
  
  if [ $? -ne 0 ]; then
    echo "Failed ori run"
    exit 1
  fi
  
  # desc
  "$PYTHON" -m methods.lift.evaluation.main \
    --task=test \
    --subtask=desc \
    --logdir="$LOG_DIR" \
    --test_img_file=toto.png \
    --test_out_file=toto.txt \
    --test_kp_file=toto.txt \
    "$AUG_OPTIONS"
  
  if [ $? -ne 0 ]; then
    echo "Failed des run"
    exit 1
  fi
  
  if ! [ -d res/lift/"$lift_data_id" ]; then
    mkdir -p res/lift/"$lift_data_id"
  fi
  
  mv kp res/lift/"$lift_data_id"/kp_aug
  mv ori res/lift/"$lift_data_id"/ori_aug
  mv des res/lift/"$lift_data_id"/des_aug

fi


if [ 1 -eq 1 ]; then
  LOG_DIR='meta/weights/lift/release-no-aug/'
  AUG_OPTIONS='--use_batch_norm=False --mean_std_type=dataset'
  
  # kp: run OK
  "$PYTHON" -m methods.lift.evaluation.main --task=test \
    --subtask=kp \
    --logdir="$LOG_DIR" \
    --test_img_file=toto.png \
    --test_out_file=toto.txt \
    "$AUG_OPTIONS"
  
  if [ $? -ne 0 ]; then
    echo "Failed kp run"
    exit 1
  fi
  
  # ori: run OK
  "$PYTHON" -m methods.lift.evaluation.main \
    --task=test --subtask=ori \
    --logdir="$LOG_DIR" \
    --test_img_file=toto.png \
    --test_out_file=toto.txt \
    --test_kp_file=toto.txt \
    "$AUG_OPTIONS"
  
  if [ $? -ne 0 ]; then
    echo "Failed ori run"
    exit 1
  fi
  
  # desc
  "$PYTHON" -m methods.lift.evaluation.main \
    --task=test \
    --subtask=desc \
    --logdir="$LOG_DIR" \
    --test_img_file=toto.png \
    --test_out_file=toto.txt \
    --test_kp_file=toto.txt \
    "$AUG_OPTIONS"
  
  if [ $? -ne 0 ]; then
    echo "Failed des run"
    exit 1
  fi
  

  if ! [ -d res/lift/"$lift_data_id" ]; then
    mkdir -p res/lift/"$lift_data_id"
  fi

  mv kp res/lift/"$lift_data_id"/kp_no_aug
  mv ori res/lift/"$lift_data_id"/ori_no_aug
  mv des res/lift/"$lift_data_id"/des_no_aug
fi  
