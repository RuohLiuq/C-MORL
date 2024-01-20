import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

from torch.distributions import Beta, Normal


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, obj_num=1):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        base = MOMLPBase
        base_kwargs['obj_num'] = obj_num
        num_outputs = action_space.shape[0]
        self.base = base(obs_shape[0], num_outputs, **base_kwargs)
        


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_mean, action_logstd = self.base(inputs, rnn_hxs, masks) 
        std = torch.exp(action_logstd)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(actor_mean, std)  # Get the Gaussian distribution

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_mean, action_logstd = self.base(inputs, rnn_hxs, masks)
        std = torch.exp(action_logstd)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(actor_mean, std)  # Get the Gaussian distribution
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=64, layernorm=True, obj_num=2):
        super(Actor, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.is_disc_action = False
        self.actor_fc1 = nn.Linear(num_inputs, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc3 = nn.Linear(hidden_size, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs)) #???


        if layernorm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    
    def forward(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd
    
    def get_kl(self, states):
        mean1, log_std1 = self.forward(states)
        std1 = torch.exp(log_std1)
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        #pdb.set_trace()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_fim(self, states):
        #pdb.set_trace()
        mean, _ = self.forward(states)
        #vec of len = No. of states*size of action e.g. cov_inv.shape = 2085*6
        cov_inv = self.actor_logstd.exp().pow(-2).squeeze(0).repeat(states.size(0)) 
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "actor_logstd":   #???
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        #pdb.set_trace()
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}

class Critic(nn.Module):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=64, layernorm=True, obj_num=2):
        super(Critic, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, obj_num)

        if layernorm:
            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

class MOMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, recurrent=False, hidden_size=64, layernorm=True, obj_num=2):
        super(MOMLPBase, self).__init__()
        self.actor = Actor(num_inputs, num_outputs, recurrent=False, hidden_size=64, layernorm=True, obj_num=obj_num)
        self.critic = Critic(num_inputs, num_outputs, recurrent=False, hidden_size=64, layernorm=True, obj_num=obj_num)
        self._hidden_size = hidden_size
        self._recurrent = recurrent

    def forward(self, states, rnn_hxs, masks):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self.actor(states)
        critic_value = self.critic(states)
        return critic_value, action_mean, action_logstd

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size