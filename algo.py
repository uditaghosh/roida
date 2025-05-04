import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
LOG_PI_NORM_MAX = 10
LOG_PI_NORM_MIN = -20

EPS = 1e-7


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

    def _get_outputs(self, state):
        a1 = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a1))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
            )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state):
        a_dist, a_tanh_mode = self._get_outputs(state)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, action):
        a_dist, _ = self._get_outputs(state)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class roida(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discriminator,
                 alpha = 7.5,
                 discount=0.5,
                 tau=0.005,
                 eta_ql=0.01,
                 alpha_wt=0.01,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=3,
                 lr=3e-4,
                 lr_decay=False,
                 critic_start_after=5000,
                 reward_thresh=0.73,
                 ):
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=0.005)

        self.discriminator = discriminator.to(device)
        self.alpha = alpha
        
        self.lr_decay = lr_decay
        
        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        critic_1 = Critic(state_dim, action_dim).to(device)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=lr)
        critic_2 = Critic(state_dim, action_dim).to(device)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=lr)

        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer
        self.critic_start_after = critic_start_after
        
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.reward_thresh = reward_thresh
        self.tau = tau
        self.eta = eta_ql
        self.alpha_wt = alpha_wt
        self.device = device
        self.step = 0
    
    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, batch_size=256, log_writer=None):

        self.step += 1

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': [], 'p_loss': [], 'wt_bc_loss': []}
        replay_buffer_e, replay_buffer_o = replay_buffer
        
        state_e, action_e, next_state_e, reward_e, not_done_e, _ = replay_buffer_e.sample_w_reward_one(batch_size)
        reward_e = reward_e * 0.9
        reward_e = -torch.log(torch.div(1-reward_e, reward_e))


        state_o, action_o, next_state_o, _, not_done_o, _ = replay_buffer_o.sample(batch_size//2)
        outs = self.discriminator(state_o,action_o)
        reward_o = -torch.log(torch.div(1-outs, outs))

        
        state = torch.cat([state_e, state_o], 0)
        action = torch.cat([action_e, action_o], 0)
        next_state = torch.cat([next_state_e, next_state_o], 0)
        reward = torch.cat([reward_e, reward_o], 0)
        not_done = torch.cat([not_done_e, not_done_o], 0)
        
        """ Q Training """
        with torch.no_grad():
            perturbed_actions, _, _ = self.ema_model(next_state)

            next_action = perturbed_actions.clamp(-self.max_action, self.max_action)


            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q
        
        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        metric['critic_loss'].append(critic_loss.item())
        
        """ Policy Training """
        if self.step % self.update_ema_every == 0:

            log_pi_e = self.actor.get_log_density(state_e, action_e)
            log_pi_o = self.actor.get_log_density(state_o, action_o)


            bc_loss = -torch.sum(log_pi_e, 1)
            lmbda = 1. / bc_loss.abs().mean().detach()
            with torch.no_grad():
                reward_o_mod_ind = torch.where(outs >= self.reward_thresh)
            
            state_o_trunc = state_o[reward_o_mod_ind[0],:]
            action_o_trunc = action_o[reward_o_mod_ind[0],:]
            reward_o_trunc = reward_o[reward_o_mod_ind[0]]

            bc_loss_only = torch.mean(bc_loss) * lmbda

            if len(reward_o_mod_ind[0]) == 0:
                wt_bc_only_loss = torch.Tensor([0])
                p_loss = self.alpha * bc_loss_only
            else:
                log_pi_o = self.actor.get_log_density(state_o_trunc, action_o_trunc)
                wt_bc_loss = -torch.sum(log_pi_o, 1) * reward_o_trunc[0]
                lmbda2 = 1. / (wt_bc_loss.abs().mean().detach())
                wt_bc_only_loss = torch.mean(wt_bc_loss) * lmbda2
                scale = float(reward_o_trunc.shape[0]) / (batch_size/2)
                wt_bc_only_loss = torch.mean(wt_bc_only_loss) * scale * self.alpha_wt
                p_loss = self.alpha * bc_loss_only + wt_bc_only_loss


            _,_,new_action = self.actor(state)
            q1_new_action = self.critic_1(state, new_action)

            lmbda3 = 1. / q1_new_action.abs().mean().detach()
            q_loss =  -lmbda3 * q1_new_action.mean()
            
            if self.critic_start_after < self.step:
                actor_loss = p_loss + self.eta * q_loss
            else:
                actor_loss = p_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.step_ema()

            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            metric['actor_loss'].append(actor_loss.item())
            metric['p_loss'].append(p_loss.item())
            metric['bc_loss'].append(bc_loss_only.item())
            metric['ql_loss'].append(q_loss.item())
            metric['wt_bc_loss'].append(wt_bc_only_loss.item())
        return metric

    def save(self, filename):
        
        torch.save(self.actor.state_dict(), filename + "_policy")
        torch.save(self.actor_optimizer.state_dict(), filename + "_policy_optimizer")

    def load(self, filename):
        
        self.actor.load_state_dict(torch.load(filename + "_policy"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))