#!/bin/bash
for env in "hopper-expert-v2" "halfcheetah-expert-v2" "walker2d-expert-v2" "ant-expert-v2"; do
unlabeled_quality='random'
seed=0

GPU_LIST=(0)

exp_num=5

for exp_unlabeled in 0 3 5; do

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python discriminate_expert_subexpert.py --env "$env" \
  --unlabeled_quality "$unlabeled_quality" --exp_num "$exp_num" --seed "$seed" \
  --exp_unlabeled "$exp_unlabeled"

done
done


for env in "pen-expert-v1" "door-expert-v1" "hammer-expert-v1" "relocate-expert-v1"; do

unlabeled_quality='cloned'
seed=0

GPU_LIST=(0)

exp_num=50

for exp_unlabeled in 0 30 50; do


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}

seed=0

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python discriminate_expert_subexpert.py --env "$env" \
  --unlabeled_quality "$unlabeled_quality" --seed "$seed"  --exp_num "$exp_num" --exp_unlabeled "$exp_unlabeled" \

done
done