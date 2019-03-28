#!/bin/sh

if [ $# -eq 0 ]; then
  echo "Usage"
  echo "  1. method"
  echo "  2. data {hpatch, strecha, webcam}"
  echo "  3. trials "
  exit 0
fi

if [ $# -gt 4 ]; then
  echo "Bad number of arguments"
  exit 1
fi

method="$1"
data="$2"
trials="$3"

log_dir="res/"$method"/"$trials"/"
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

# evaluation of cv detector and descriptor
ok=1
if [ "$ok" -eq 1 ]; then 
  python3 -m methods.cv."$data" \
    --method "$method" \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1
fi



# evaluation of cv detector with proxy descriptor
ok=0
if [ "$ok" -eq 1 ]; then
  python3 -m methods.cv."$data"_des_proxy \
    --method "$method" \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1 \
    --feat_name pool3
fi

# evaluation of elf detector with cv descriptor
ok=0
if [ "$ok" -eq 1 ]; then

    python3 -m methods.cv."$data"_det_elf \
      --method "$method" \
      --trials "$trials" \
      --h 480 \
      --w 640 \
      --nms_dist 10 \
      --border_remove 10 \
      --max_num_feat 500 \
      --trial "$trials" \
      --grad_name pool2 \
      --feat_name pool3 \
      --thr_k_size 5 \
      --thr_sigma 5 \
      --noise_k_size 5 \
      --noise_sigma 4 \
      --resize 1

fi
