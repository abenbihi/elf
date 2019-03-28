#!/bin/sh

export PYTHONPATH=$PYTHONPATH:/home/ws/methods/lfnet/lf-net-release/
LOG_DIR=res/elf-lfnet

if [ $# -eq 0 ]; then
  echo "Usage"
  echo "  1. trials "
  exit 0
fi

if [ $# -ne 1 ]; then
  echo "Bad number of arguments"
  exit 1
fi


trials=$1
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

grad_block=2
grad_name=conv
feat_block=3
feat_name=pool3

python3 -m methods.lfnet.elf.hpatch \
  --h 480 \
  --w 640 \
  --resize 1 \
  --nms_dist 10 \
  --border_remove 10 \
  --max_num_feat 500 \
  --trials "$trials" \
  --kp_size 30 \
  --grad_block "$grad_block" \
  --grad_name "$grad_name" \
  --feat_block "$feat_block" \
  --feat_name "$feat_name" \
  --thr_k_size 5 \
  --thr_sigma 5 \
  --noise_k_size 5 \
  --noise_sigma 4 \
  --model ./meta/weights/lfnet/outdoor/ \


