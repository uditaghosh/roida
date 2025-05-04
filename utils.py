import numpy as np
import torch
import pickle


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.flag = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_dataset(self, path, no_normalize=False):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        self.state = dataset['observations']
        self.action = dataset['action']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards']
        self.not_done = dataset['not_done']
        self.flag = dataset['flag']
        self.size = dataset['size']
        if not no_normalize:
            self.mean = dataset['mean']
            self.std = dataset['std']

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.flag[ind]).to(self.device),
        )
    
    def sample_w_reward_one(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(np.ones_like(self.reward[ind])).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.flag[ind]).to(self.device),
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.flag = dataset['flag'].reshape(-1, 1)
        self.size = self.state.shape[0]
        return {'observations': self.state,
                'action': self.action,
                'next_observations': self.next_state,
                'rewards': self.reward,
                'not_done': self.not_done,
                'flag': self.flag,
                'size':self.size}

    def normalize_states(self, eps=1e-3, mean=None, std=None):
        if mean is None and std is None:
            mean = self.state.mean(0, keepdims=True)
            std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std, {'normalized_state': self.state, 'normalized_next_state': self.next_state, 'mean': mean, 'std': std}
