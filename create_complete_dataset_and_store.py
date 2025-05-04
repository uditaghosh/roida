import numpy as np
import pickle
import torch
import gym
import argparse
import os
import d4rl
import time
import matplotlib.pyplot as plt

import utils
import get_dataset


if __name__ == "__main__":#incorporate entire algorithm
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--root_dir", default="new_results")
    parser.add_argument('--env', default="hopper-expert-v2")
    parser.add_argument("--split_x", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--unlabeled_quality", default='random')
    parser.add_argument("--no_normalize", action='store_true')
    parser.add_argument("--exp_num", default=5, type=int)
    parser.add_argument("--exp_unlabeled", default=0, type=int)
    args = parser.parse_args()

dataset_name = f"{args.env}_{args.split_x}_{args.unlabeled_quality}_{args.exp_num}_{args.exp_unlabeled}"

os.makedirs(f"{args.root_dir}/{dataset_name}", exist_ok=True)
save_dir = f"{args.root_dir}/{dataset_name}"
print("---------------------------------------")
print(f"Dataset: {dataset_name}")
print("---------------------------------------")

env_e = gym.make(args.env)
env_id = args.env.split('-')[0]
if env_id in {'hopper', 'walker2d', 'ant'}:
    if args.unlabeled_quality == 'random':
        env_o = gym.make(f'{env_id}-random-v2')
    exp_num = args.exp_num + args.exp_unlabeled
    args.split_x = ((1.0 * args.exp_unlabeled) / exp_num ) * 100.0
    print(args.split_x)
else:
    if args.unlabeled_quality == 'cloned':
        env_o = gym.make(f'{env_id}-cloned-v1')
    exp_num = args.exp_num + args.exp_unlabeled
    args.split_x = ((1.0 * args.exp_unlabeled) / exp_num ) * 100.0
    print(args.split_x)

# Set seeds
env_e.seed(args.seed)
env_e.action_space.seed(args.seed)
env_o.seed(args.seed)
env_o.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env_e.observation_space.shape[0]
action_dim = env_e.action_space.shape[0]

dataset_e_raw = env_e.get_dataset()
dataset_o_raw = env_o.get_dataset()
split_x = int(np.round(exp_num * args.split_x / 100))
dataset_e, dataset_o = get_dataset.dataset_setting1(dataset_e_raw, dataset_o_raw, split_x, exp_num)
dataset_b = get_dataset.dataset_concat(dataset_e, dataset_o)

states_e = dataset_e['observations']
states_o = dataset_o['observations']
states_b = np.concatenate([states_e, states_o]).astype(np.float32)

print('# {} of expert demonstraions'.format(states_e.shape[0]))
print('# {} of imperfect demonstraions'.format(states_o.shape[0]))

replay_buffer_e = utils.ReplayBuffer(state_dim, action_dim)
replay_buffer_o = utils.ReplayBuffer(state_dim, action_dim)
replay_buffer_b = utils.ReplayBuffer(state_dim, action_dim)
dict_e = replay_buffer_e.convert_D4RL(dataset_e)
dict_o = replay_buffer_o.convert_D4RL(dataset_o)
dict_b = replay_buffer_b.convert_D4RL(dataset_b)

_, _, dict_e_norm = replay_buffer_e.normalize_states()
_, _, dict_o_norm = replay_buffer_o.normalize_states()
mean, std, dict_b_norm = replay_buffer_b.normalize_states()

print('mean:', mean)
print('std:', std)

with open(os.path.join(save_dir, 'expert_data.pkl'), 'wb') as f_e:
    pickle.dump({**dict_e, **dict_e_norm}, f_e)

with open(os.path.join(save_dir, 'unlabeled_data.pkl'), 'wb') as f_o:
    pickle.dump({**dict_o, **dict_o_norm}, f_o)

with open(os.path.join(save_dir, 'combined_expert_and_unlabled_data.pkl'), 'wb') as f_b:
    pickle.dump({**dict_b, **dict_b_norm}, f_b)