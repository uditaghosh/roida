#import pickle
import os
import gym
import numpy as np
import torch
import d4rl
from utils import ReplayBuffer
from discriminator import DiscriminatorCritic, Discriminator
import argparse
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt


def parse_list_from_string(list_string):
    # Split the string into a list
    my_list = list_string.split(',')
    my_list = [int(i) for i in my_list]
    return my_list

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Env: {env_name}, Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return avg_reward, d4rl_score

def worker(seed, args):
    import algo
    file = f"/home/csgrad/ughosh/imitationL/dwbc_w_q_learning/datasets/{args.env}_{args.split_x}_{args.unlabeled_quality}_{args.exp_num}_{args.exp_unlabeled}"
    dataset_name = f"{args.env}-{args.split_x}-{args.unlabeled_quality}_{args.exp_num}_{args.exp_unlabeled}"
    algo_name = f"{args.algorithm}"
    algo_specific_details = None

    os.makedirs(f"{args.root_dir}/{dataset_name}/{algo_name}/{algo_specific_details}/models", exist_ok=True)
    save_model_dir = f"{args.root_dir}/{dataset_name}/{algo_name}/{algo_specific_details}/models"
    save_dir = f"{args.root_dir}/{dataset_name}/{algo_name}/{algo_specific_details}/seed-{seed}.txt"

    discriminator_path = os.path.join(args.discriminator_dir, f"{args.env}_{args.split_x}_{args.unlabeled_quality}_{args.exp_num}_{args.exp_unlabeled}/models/model_1")

    env_e = gym.make(args.env)
    env_id = args.env.split('-')[0]
    if env_id in {'hopper', 'walker2d', 'ant'}:
        env_o = gym.make(f'{env_id}-random-v2')
    else:
        env_o = gym.make(f'{env_id}-cloned-v1')

    # Set seeds
    env_e.seed(seed)
    env_e.action_space.seed(seed)
    env_o.seed(seed)
    env_o.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env_e.observation_space.shape[0]
    action_dim = env_e.action_space.shape[0]
    max_action = float(env_e.action_space.high[0])


    replay_buffer_e = ReplayBuffer(state_dim, action_dim)
    replay_buffer_o = ReplayBuffer(state_dim, action_dim)
    replay_buffer_b = ReplayBuffer(state_dim, action_dim)
    replay_buffer_e.load_dataset(os.path.join(file, 'expert_data.pkl'), no_normalize=True)
    replay_buffer_o.load_dataset(os.path.join(file, 'unlabeled_data.pkl'), no_normalize=True)
    replay_buffer_b.load_dataset(os.path.join(file, 'combined_expert_and_unlabled_data.pkl'), no_normalize=True)

    if args.no_normalize:
        shift, scale = 0, 1
    else:
        shift, scale = None, None

        shift, scale, _ = replay_buffer_b.normalize_states()
        _ = replay_buffer_e.normalize_states(mean=shift, std=scale)
        _ = replay_buffer_o.normalize_states(mean=shift, std=scale)


    eval_log = open(save_dir, 'w')

    discriminator = Discriminator(state_dim,action_dim)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    discriminator.load_state_dict(torch.load(discriminator_path + "_discriminator"))
    discriminator_optimizer.load_state_dict(torch.load(discriminator_path + "_discriminator_optimizer"))
  
    # Initialize policy
    if args.algorithm == 'ROIDA':
        policy = algo.roida(state_dim, action_dim, max_action, discriminator, discount=args.discount, eta_ql=args.eta_ql, alpha_wt=aegs.alpha_wt, lr=args.lr)


    score = []

    for t in range(int(args.max_timesteps)):
        _ = policy.train([replay_buffer_e, replay_buffer_o], batch_size=args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            average_reward, d4rl_score = eval_policy(policy, args.env, seed, shift, scale)
            score.append(d4rl_score)
            eval_log.write(f'{t + 1}\t{d4rl_score}\t\t{average_reward}\t\n')
            eval_log.flush()
    policy.save(os.path.join(save_model_dir, f"seed_{seed}"))
    eval_log.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--root_dir", default="results")
    parser.add_argument("--algorithm", default="ROIDA")
    parser.add_argument('--env', default="hopper-expert-v2") 
    parser.add_argument("--split_x", default=0, type=int)
    parser.add_argument("--unlabeled_quality", default='random')
    parser.add_argument("--num_seeds", default=0, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=5e5, type=int)
    
    parser.add_argument("--discriminator_dir", default='discriminator_results')
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--alpha", default=7.5, type=float)
    parser.add_argument("--eta", default=0.5, type=float)
    parser.add_argument("--eta_ql", default=0.1, type=float)
    parser.add_argument("--alpha_wt", default=0.1, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--no_normalize", action='store_true')
    parser.add_argument("--exp_num", default=5, type=int)
    parser.add_argument("--exp_unlabeled", default=0, type=int)
    parser.add_argument("--seed_list", default='0,1,2,3,4')
    args = parser.parse_args()

    seeds = parse_list_from_string(args.seed_list)
    partial_worker = partial(worker, args=args)

    mp.set_start_method('spawn')
    with mp.Pool(processes=args.num_seeds) as pool:
        pool.map(partial_worker, seeds)