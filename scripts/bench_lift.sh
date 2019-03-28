#!/bin/sh


export PYTHONPATH=$PYTHONPATH:/home/ws/methods/lift/tf-lift

PYTHON=python3

if [ $# -eq 0 ]; then
  echo "Usage"
  echo "  1. lift data id (lift kp and des are stored in res/lift/id)"
  echo "  2. data {hpatch, strecha, webcam}"
  echo "  3. trials "
  echo "  4. elf kp dir id (optional) (elf kp are stored in res/elf/id)"
  exit 0
fi

if [ $# -gt 4 ]; then
  echo "Bad number of arguments"
  exit 1
fi

lift_data_id="$1"
data="$2"
trials="$3"

log_dir="res/lift/"$trials"/"

# evaluation of lift detector and descriptor
ok=1
if [ "$ok" -eq 1 ]; then 

  python3 -m methods.lift.evaluation."$data" \
    --lift_data_id "$lift_data_id" \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1
fi

# evaluation of lift detector with proxy descriptor
ok=0
if [ "$ok" -eq 1 ]; then
  if ! [ "$data" = 'hpatch' ]; then 
    echo "Sorry, this mode is currently available with hpatch only"
    exit 1
  fi

  python3 -m methods.lift.elf."$data"_des_proxy \
    --lift_data_id "$lift_data_id" \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1 \
    --feat_name pool3
fi

# evaluation of elf detector with lift descriptor
ok=0
if [ "$ok" -eq 1 ]; then

  
  if [ $# -ne 4 ]; then
    echo "Bad number of arguments: please specify the kp dir."
    exit 1
  fi

  if ! [ "$data" = 'hpatch' ]; then 
    echo "Sorry, this mode is currently available with hpatch only"
    exit 1
  fi

  elf_kp_dir_id="$4"

  # generate elf kp
  python3 -m methods.lift.elf."$data"_kp_elf \
    --H 480 \
    --W 640 \
    --nms_dist 10 \
    --border_remove 10 \
    --max_num_feat 500 \
    --trials "$elf_kp_dir_id" \
    --grad_name pool2 \
    --feat_name pool4 \
    --thr_k_size 5 \
    --thr_sigma 5 \
    --noise_k_size 5 \
    --noise_sigma 4 \
    --model vgg \
    --resize 1 \
    --save2txt 1

  if [ $? -ne 0 ]; then 
      echo "Error during ELF keypoint generations. Abort."
      exit 1
  fi

  ############################################################################ 
  # gen lift descriptor

  # rotation augmentation for v_ images
  LOG_DIR='meta/weights/lift/release-aug/' # path to trained weights
  AUG_OPTIONS='--use_batch_norm=False --mean_std_type=hardcoded'
  
  # kp: run OK
  "$PYTHON" -m methods.lift.evaluation.main --task=test \
    --elf_kp=1 \
    --kp_dir_id="$elf_kp_dir_id" \
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
  else
      rm res/lift/"$lift_data_id"
      mkdir -p res/lift/"$lift_data_id"
  fi
  
  mv kp res/lift/"$lift_data_id"/kp_aug
  mv ori res/lift/"$lift_data_id"/ori_aug
  mv des res/lift/"$lift_data_id"/des_aug

 
  # ##########################################################################
  # model without augmentation
  LOG_DIR='meta/weights/lift/release-no-aug/'
  AUG_OPTIONS='--use_batch_norm=False --mean_std_type=dataset'
  
  # kp: run OK
  "$PYTHON" -m methods.lift.evaluation.main --task=test \
     --elf_kp=1 \
    --kp_dir_id="$elf_kp_dir_id" \
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


  ############################################################################## 
  # evaluate them
  python3 -m methods.lift.evaluation."$data" \
    --lift_data_id "$lift_data_id" \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1

fi
