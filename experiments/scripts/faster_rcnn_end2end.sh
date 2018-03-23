#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

DEV='cpu'
DEV_ID=0
NET=VGG16
DATASET=pascal_voc

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="/home/wjd/python/1/Faster-RCNN_TF-master/data/CAR2017"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=1
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="/home/wjd/python/1/Faster-RCNN_TF-master/experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python /home/wjd/python/1/Faster-RCNN_TF-master/tools/train_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights /home/wjd/python/1/Faster-RCNN_TF-master/data/pretrain_model/VGG_imagenet.npy \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg /home/wjd/python/1/Faster-RCNN_TF-master/experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_train \
  ${EXTRA_ARGS}

#set +x
#NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
#NET_FINAL=/home/wjd/python/1/Faster-RCNN_TF-master/output/faster_rcnn_end2end/voc_2007_trainval/VGGnet_fast_rcnn_iter_1.ckpt
#set -x

#time python /home/wjd/python/1/Faster-RCNN_TF-master/tools/test_net.py --device ${DEV} --device_id ${DEV_ID} \
 # --weights ${NET_FINAL} \
 # --imdb ${TEST_IMDB} \
 # --cfg /home/wjd/python/1/Faster-RCNN_TF-master/experiments/cfgs/faster_rcnn_end2end.yml \
#  --network VGGnet_test \
#  ${EXTRA_ARGS}