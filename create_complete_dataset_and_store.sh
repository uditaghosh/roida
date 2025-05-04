#!/bin/bash
for env in "hopper-expert-v2" "halfcheetah-expert-v2" "walker2d-expert-v2" "ant-expert-v2"; do

exp_num=5

GPU_LIST=(0)

for exp_unlabeled in 0 3 5; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python create_complete_dataset_and_store_unlabeled.py --root_dir datasets --env $env \
 --split_x 0 --seed 1 --unlabeled_quality random \
 --exp_num $exp_num --exp_unlabeled $exp_unlabeled


done
done

for env in "pen-expert-v1" "door-expert-v1" "hammer-expert-v1" "relocate-expert-v1"; do

exp_num=50

GPU_LIST=(0)

for exp_unlabeled in 0 30 50; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python create_complete_dataset_and_store_unlabeled.py --root_dir datasets --env $env \
 --split_x 0 --seed 1 --unlabeled_quality cloned \
 --exp_num $exp_num --exp_unlabeled $exp_unlabeled


done
done
