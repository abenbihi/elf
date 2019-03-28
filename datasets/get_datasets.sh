#!/bin/sh

# function copied from https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# Hpatches
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xvzf hpatches-sequences-release.tar.gz
rm hpatches-sequences-release.tar.gz


# Webcam
wget https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/keypoints/WebcamRelease.tar.gz
tar -xvzf WebcamRelease.tar.gz
rm WebcamRelease.tar.gz


# Hpatches scale
fileid=1hEaQclVvAFz_H4EEm2Ak48tnkagaYgTA
filename=hpatches_s.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"
rm "$filename"


# Hpatches rotation
fileid=1jLoIP00pDPRqrc-uRyyxsGzpAF2yiEKM
filename=hpatches_rot.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"
rm "$filename"


# Strecha
fileid=124Qfwmthh4Wj3n_kAxEb7rVg2eEN6gBj
filename=strecha.tar.gz
gdrive_download "$fileid" "$filename"
tar -xvzf "$filename"
rm "$filename"



