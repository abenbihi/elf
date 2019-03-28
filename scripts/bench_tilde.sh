#!/bin/sh

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

trials="$1"
data="$2"

if [ "$data" = "hpatch" ]; then
  
  # gen kp
  while read -r line
  do
    scene="$line"
    mkdir res/tilde/"$trials"/"$scene"/wo -p
  done < meta/list/img_hp.txt
  
  ./methods/tilde/c++/build/HPatch/demo_hpatch "$trials"

  if [ "$?" -ne 0 ]; then
    echo "Error during tilde keypoint generation. Abort"
    exit 1
  fi
  
  # gen desriptor + evaluation
  python3 -m methods.tilde.hpatch \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1 \
    --feat_name pool3
fi

if [ "$data" = "webcam" ]; then
  
  # gen kp
  while read -r line
  do
    scene="$line"
    mkdir res/tilde/"$trials"/"$scene"/wo -p
  done < meta/list/img_webcam.txt
  
  ./methods/tilde/c++/build/Webcam/demo_webcam "$trials"
 
  if [ "$?" -ne 0 ]; then
    echo "Error during tilde keypoint generation. Abort"
    exit 1
  fi
  
  # gen desriptor + evaluation
  python3 -m methods.tilde.webcam \
    --trials "$trials" \
    --max_num_feat 500 \
    --h 480 \
    --w 640 \
    --resize 1 \
    --feat_name pool3
fi


