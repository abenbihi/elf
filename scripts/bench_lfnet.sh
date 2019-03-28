#!/bin/sh

LOG_DIR=res/lfnet/

export PYTHONPATH=$PYTHONPATH:/home/ws/methods/lfnet/lf-net-release/

if [ $# -eq 0 ]; then
  echo "  1. trials"
  echo "  2. data"
  echo "  3. model {indoor, outdoor}"
  echo "  4. kp data id (optional)"
  exit 0
fi

if [ $# -gt 4 ]; then
  echo "Bad number of arguments"
  exit 1
fi


trials="$1"
data="$2"
model="$3"


log_dir=""$LOG_DIR"/"$trials"/"
if [ -d "$log_dir" ]; then
  while true; do
    read -p ""$log_dir" already exists. Do you want to overwrite it (y/n) ?" yn
    case $yn in
      [Yy]* ) rm -rf "$log_dir"; mkdir -p "$log_dir"; break;;
      [Nn]* ) exit;;
      * ) * echo "Please answer yes or no.";;
    esac
  done
else
  mkdir -p "$log_dir"
fi

##############################################################################
# evaluation of lfnet detector and descriptor
ok=0
if [ "$ok" -eq 1 ]; then 

  python3 -m methods.lfnet.evaluation."$data" \
    --trials "$trials" \
    --model ./meta/weights/lfnet/"$model"/ \
    --use_nms3d False \
    --h 480 \
    --w 640 \
    --resize 1
fi


##############################################################################
# evaluation of lfnet detector with proxy descriptor
ok=0
if [ "$ok" -eq 1 ]; then


  if [ $# -ne 4 ]; then
    echo "Bad number of arguments: please specify the kp dir."
    exit 1
  fi
  kp_dir_id="$4"


  if ! [ "$data" = 'hpatch' ]; then 
    echo "Sorry, this mode is currently available with hpatch only"
    exit 1
  fi

  # lfnet detection
  python3 -m methods.lfnet.evaluation."$data" \
    --trials "$kp_dir_id" \
    --model ./meta/weights/lfnet/"$model"/ \
    --use_nms3d False \
    --h 480 \
    --w 640 \
    --resize 1 \
    --save2txt 1

  if [ $? -ne 0 ]; then 
      echo "Error during lfnet keypoint generations. Abort."
      exit 1
  fi

  # proxy description
  python3 -m methods.lfnet.elf."$data"_des_proxy \
    --kp_dir_id "$kp_dir_id" \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1 \
    --feat_name pool3

fi

##############################################################################
# evaluation of elf detector with lfnet descriptor
ok=1
if [ "$ok" -eq 1 ]; then

  if [ $# -ne 4 ]; then
    echo "Bad number of arguments: please specify the kp dir."
    exit 1
  fi
  kp_dir_id="$4"

  if ! [ "$data" = 'hpatch' ]; then 
    echo "Sorry, this mode is currently available with hpatch only"
    exit 1
  fi

  # generate elf kp
  python3 -m methods.lfnet.elf."$data"_kp_elf \
    --H 480 \
    --W 640 \
    --nms_dist 10 \
    --border_remove 10 \
    --max_num_feat 500 \
    --trials "$kp_dir_id" \
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

  # lfnet description
  python3 -m methods.lfnet.elf."$data"_det_elf \
    --trials "$trials" \
    --kp_dir_id "$kp_dir_id" \
    --model ./meta/weights/lfnet/"$model"/ \
    --use_nms3d False \
    --h 480 \
    --w 640 \
    --resize 1
fi










