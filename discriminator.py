import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
LOG_PI_NORM_MAX = 10
LOG_PI_NORM_MIN = -20

EPS = 1e-7

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.fc1_1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        d1 = F.relu(self.fc1_1(sa))
        d = F.relu(self.fc2(d1))
        d = F.relu(self.fc3(d))
        d = F.sigmoid(self.fc4(d))
        d = torch.clip(d, 0.1, 0.9) #not sure whether to use clip or not
        return d

class DiscriminatorCritic(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 no_pu=False,
                 eta=0.5,
    ):
        self.discriminator = Discriminator_1(state_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.discriminator_optimizer, T_max=5e3)
        self.no_pu_learning = no_pu
        self.eta = eta

    def train(self, replay_buffer_e, replay_buffer_o, batch_size=256, epoch=5e4):

        metric = {'d_loss': [],}
        
        for _ in tqdm(range(int(epoch))):
            state_e, action_e, _, _, _, flag_e = replay_buffer_e.sample(batch_size)
            state_o, action_o, _, _, _, flag_o = replay_buffer_o.sample(batch_size)
            d_e = self.discriminator(state_e, action_e)
            d_o = self.discriminator(state_o, action_o)

            d_loss_e = -torch.log(d_e)
            d_loss_o = - F.softplus(torch.log(1 - d_o) / self.eta - torch.log(1 - d_e))
            d_loss = torch.mean(d_loss_e + d_loss_o)
            metric['d_loss'].append(d_loss.item())

            # Optimize the discriminator
            self.discriminator_optimizer.zero_grad()
            d_loss.backward()
            self.discriminator_optimizer.step()
            self.scheduler.step()
        
        return metric
    
    def output(self, state, action):
        return self.discriminator(state, action)

    def save(self, filename):
        torch.save(self.discriminator.state_dict(), filename + "_discriminator")
        torch.save(self.discriminator_optimizer.state_dict(), filename + "_discriminator_optimizer")
    
    def load(self, filename):
        self.discriminator.load_state_dict(torch.load(filename + "_discriminator"))
        self.discriminator_optimizer.load_state_dict(torch.load(filename + "_discriminator_optimizer"))
