#!/bin/bash

SOURCE="/home/zhangniansong/imagenet_calib/calibration.txt"
ROOT="/home/zhangniansong/imagenet_calib/img/"
INPUT_SIZE=224
IMAGENET="/home/SharedDatasets/imagenet"
GPU=0
# ARCH="proxyless_mobile_14"
ARCH="proxyless_mobile_05"

python export_caffe.py \
    -p $IMAGENET \
    -g $GPU \
    -a $ARCH \
    -s $SOURCE \
    -r $ROOT \
    -d $INPUT_SIZE