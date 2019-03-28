#!/bin/sh


if [ $# -eq 0 ]; then
  echo "Usage"
  echo "  1. trials "
  echo "  2. data {hpatch, strecha, webcam}"
  exit 0
fi

if [ $# -ne 2 ]; then
  echo "Bad number of arguments"
  exit 1
fi


trials=$1
data="$2"

log_dir=res/elf/"$trials"/
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


# my perf
python3 -m methods.elf_vgg."$data" \
  --H 480 \
  --W 640 \
  --nms_dist 10 \
  --border_remove 10 \
  --max_num_feat 500 \
  --trial "$trials" \
  --grad_name pool2 \
  --feat_name pool3 \
  --thr_k_size 17 \
  --thr_sigma 6 \
  --noise_k_size 5 \
  --noise_sigma 4 \
  --model vgg \
  --resize 1


