#!/bin/sh

LOG_DIR=res/superpoint/

if [ $# -eq 0 ]; then
  echo "Usage"
  echo "  1. trials "
  echo "  2. data {hpatch, webcam}"
  exit 0
fi

if [ $# -ne 2 ]; then
  echo "Bad number of arguments"
  exit 1
fi

trials=$1
data="$2"
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

grad_name=conv1a
feat_name=pool3

python3 -m methods.elf_superpoint."$data" \
  --trials "$trials" \
  --H 480 \
  --W 640 \
  --nms_dist 10 \
  --border_remove 10 \
  --max_num_feat 500 \
  --trials "$trials" \
  --grad_name "$grad_name" \
  --feat_name "$feat_name" \
  --thr_k_size 19 \
  --thr_sigma 7 \
  --noise_k_size 7 \
  --noise_sigma 2 \
  --resize 1

if [ "$?" -ne 0 ]; then
  echo "Error in run "$trials""
  exit 1
fi



