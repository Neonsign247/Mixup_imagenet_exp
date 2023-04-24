#!/bin/bash
NUM_PROC=$1
shift
echo CUDA_VISIBLE_DEVICES=$NUM_PROC 
export CUDA_VISIBLE_DEVICES=$NUM_PROC
for i in 0 1 2 3 4 5 6 7 8 9
do
    python main.py --dataset cifar100 --labels_per_class 500 --arch preactresnet34  \
        --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 300 \
        --schedule 100 200 --gammas 0.1 0.1 --train vanilla --seed $i
done