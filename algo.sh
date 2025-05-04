#!/bin/bash

env="hopper-expert-v2"
unlabeled_quality='random'
split_x=0

seed=()
for i in {1..5}; do
  seed+=($(( (RANDOM % 100) + 1 )))
done

list_string="(${seed[@]})"


GPU_LIST=(0)

log_folder="logs_for_"$algorithm
mkdir "$log_folder"

exp_num=5

for exp_unlabeled in 0 3 5; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env $env \
  --split_x $split_x \
  --unlabeled_quality $unlabeled_quality \
  --exp_num $exp_num \
  --exp_unlabeled $exp_unlabeled \
  --num_seed 5 \
  --seed_list "$list_string"

done
done
