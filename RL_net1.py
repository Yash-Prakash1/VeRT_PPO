import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
 

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        # dropout = nn.Dropout2d(0.2) if j < len(sizes) - 1 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(Actor, self).__init__()
        
        self.lp = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,n_actions))
        self.logstd = nn.Parameter(torch.zeros(1, n_actions))           
        

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.tensor(x[0], dtype=torch.float)
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        ot_n = self.lp(x)
        std = torch.exp(self.logstd)
        return F.tanh(ot_n)
        


class Critic(nn.Module):
    '''
    Actor neural network
    '''
    def __init__(self, input_shape):
        super(Critic, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))


    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.tensor(x[0], dtype=torch.float)
        if isinstance(x,np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.lp(x)

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim=19, act_dim=5):
        super().__init__()

        # build policy and value functions
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)
        # self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mean = self.actor(obs)
            print(mean)
            logstd = self.actor.logstd.data
            # mean = torch.tensor(mean, dtype=
            # torch.float32)
            action = mean + np.exp(logstd) * np.random.normal(size=logstd.shape)
            return action.numpy()