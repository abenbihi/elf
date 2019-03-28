#!/bin/sh

# function copied from https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# vgg
mkdir -p vgg
fileid=1ZS656akAWuFaX5SodXL1Hx1q3A-fPLp-
filename=vgg/data.ckpt
gdrive_download "$fileid" "$filename"

# alexnet
mkdir -p alexnet
fileid=14XWtYHrqKzAibtGfpLIquRrosXVOY6kA
filename=alexnet/data.ckpt
gdrive_download "$fileid" "$filename"

# xception
fileid=1bljfpxrOVZAeLyif5tpVussf_0YDgcXB
filename=xception.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"
rm "$filename"


# superpoint-tf
mkdir -p superpoint
fileid=1qd-sv-wPLKj_8t03q9NbzaPsD5f2zyqn
filename=superpoint/superpoint-tf.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"
mv npy superpoint
rm "$filename"


# superpoint
wget https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth
mv superpoint_v1.pth superpoint


# lift
wget http://webhome.cs.uvic.ca/~kyi/files/2018/tflift/release-no-aug.tar.gz
tar -xvzf release-no-aug.tar.gz
rm -f release-no-aug.tar.gz

wget http://webhome.cs.uvic.ca/~kyi/files/2018/tflift/release-aug.tar.gz
tar -xvzf release-aug.tar.gz
rm -f release-aug.tar.gz

mkdir -p lift
mv release-no-aug lift
mv release-aug lift

# lfnet
wget http://webhome.cs.uvic.ca/~kyi/files/2018/lf-net/pretrained.tar.gz
tar -xvzf pretrained.tar.gz
rm pretrained.tar.gz

mkdir -p lfnet
mv release/models/outdoor lfnet
mv release/models/indoor lfnet
