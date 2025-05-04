import numpy as np
import pickle
import torch
import gym
import argparse
import os
import d4rl
import time

import utils
import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--root_dir", default="discriminator_results")
    parser.add_argument("--dataset_dir", default="datasets")
    parser.add_argument('--env', default="hopper-expert-v2")
    parser.add_argument("--split_x", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--unlabeled_quality", default='random')
    parser.add_argument("--batch_size", default=256, type=int)
    # Discriminator
    parser.add_argument("--no_pu", action='store_true')
    parser.add_argument("--eta", default=0.5, type=float)
    parser.add_argument("--no_normalize", action='store_true')

    parser.add_argument("--exp_num", default=5, type=int)
    parser.add_argument("--exp_unlabeled", default=0, type=int)
    args = parser.parse_args()

    args.root_dir = "discriminator_results"
    from discriminator import DiscriminatorCritic

    dataset_name = f"{args.env}_{args.split_x}_{args.unlabeled_quality}_{args.exp_num}_{args.exp_unlabeled}"
    

    os.makedirs(f"{args.root_dir}/{dataset_name}/models", exist_ok=True)
    save_dir = f"{args.root_dir}/{dataset_name}/models"
    dataset_dir = f"{args.dataset_dir}/{dataset_name}"
    print("---------------------------------------")
    print(f"Dataset: {dataset_name}, Seed: {args.seed}")
    print("---------------------------------------")

    path_e = os.path.join(dataset_dir, 'expert_data.pkl')
    path_o = os.path.join(dataset_dir, 'unlabeled_data.pkl')
    path_b = os.path.join(dataset_dir, 'combined_expert_and_unlabled_data.pkl')

    model_save_path = os.path.join(save_dir, 'model_1')

    env_e = gym.make(args.env)
    env_id = args.env.split('-')[0]
    if env_id in {'hopper', 'walker2d', 'ant'}:
        env_o = gym.make(f'{env_id}-random-v2')
    else:
        env_o = gym.make(f'{env_id}-cloned-v1')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env_e.observation_space.shape[0]
    action_dim = env_e.action_space.shape[0]


    replay_buffer_e = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer_o = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer_b = utils.ReplayBuffer(state_dim, action_dim)
    
     #load dataset
    replay_buffer_e.load_dataset(path_e, args.no_normalize)
    replay_buffer_o.load_dataset(path_o, args.no_normalize)
    replay_buffer_b.load_dataset(path_b, args.no_normalize)
    max_action = float(max(env_e.action_space.high[0], env_o.action_space.high[0]))

    #train discriminator
    discriminator = DiscriminatorCritic(state_dim,
                                        action_dim,
                                        no_pu=args.no_pu,
                                        eta=args.eta)
    discriminator_metric = discriminator.train(replay_buffer_e=replay_buffer_e, replay_buffer_o=replay_buffer_o, batch_size= args.batch_size, epoch=5e4)
    discriminator.save(model_save_path)
    