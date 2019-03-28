#!/bin/sh


LOG_DIR=res/superpoint/

if [ $# -eq 0 ]; then
  echo "Usage"
  echo "  1. trials "
  echo "  2. data {hpatch, strecha, webcam}"
  echo "  3. kp dir id (optional)"
  exit 0
fi

if [ $# -ge 4 ]; then
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
            #[Yy]* ) echo "Results will be overwritten"; break ;;
            [Nn]* ) exit;;
            * ) * echo "Please answer yes or no.";;
        esac
    done
else
    mkdir -p "$log_dir"
fi


# evaluation of superpoint detector and descriptor
ok=1
if [ "$ok" -eq 1 ]; then 
  python3 -m methods.superpoint."$data" \
    --trials "$trials" \
    --weights_path meta/weights/superpoint/superpoint_v1.pth \
    --H 480 \
    --W 640 \
    --nms_dist 10 \
    --border_remove 10 \
    --max_num_features 500 \
    --trials "$trials" \
    --resize 1
fi


# evaluation of superpoint detector with proxy descriptor
ok=0
if [ "$ok" -eq 1 ]; then

  if [ $# -ne 3 ]; then
    echo "Bad number of arguments"
    exit 1
  fi
  kp_dir_id="$3"


  python3 -m methods.superpoint."$data" \
    --trials "$kp_dir_id" \
    --weights_path meta/weights/superpoint/superpoint_v1.pth \
    --H 480 \
    --W 640 \
    --nms_dist 10 \
    --border_remove 10 \
    --max_num_features 500 \
    --resize 1 \
    --save2txt 1

  python3 -m methods.superpoint."$data"_des_proxy \
    --kp_dir_id "$kp_dir_id" \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1 \
    --feat_name pool3
fi


# evaluation of elf detector with superpoint descriptor
ok=0
if [ "$ok" -eq 1 ]; then

  
  if [ $# -ne 3 ]; then
    echo "Bad number of arguments: please specify the kp dir."
    exit 1
  fi
  kp_dir_id="$3"


  if ! [ "$data" = 'hpatch' ]; then 
    echo "Sorry, this mode is currently available with hpatch only"
    exit 1
  fi


  ## generate elf kp
  python3 -m methods.elf_vgg."$data" \
    --H 480 \
    --W 640 \
    --nms_dist 10 \
    --border_remove 10 \
    --max_num_feat 500 \
    --trial "$kp_dir_id" \
    --grad_name pool2 \
    --feat_name pool4 \
    --thr_k_size 5 \
    --thr_sigma 5 \
    --noise_k_size 5 \
    --noise_sigma 4 \
    --model vgg \
    --resize 1 \
    --save2txt 1


  # describe elf kp with opencv descriptor
  python3 -m methods.superpoint."$data"_det_elf \
    --trials "$trials" \
    --kp_dir_id "$kp_dir_id" \
    --weights_path meta/weights/superpoint/superpoint_v1.pth \
    --H 480 \
    --W 640 \
    --nms_dist 10 \
    --border_remove 10 \
    --max_num_features 500 \
    --trials "$trials" \
    --resize 1

fi
