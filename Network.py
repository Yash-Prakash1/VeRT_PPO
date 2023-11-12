"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    '''
    Policy neural network
    '''
    def __init__(self, input_shape, n_actions):
        super(Actor, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,n_actions))

        # self.mean_l = nn.Linear(32, n_actions)
        # self.mean_l.weight.data.mul_(0.1)

        # self.var_l = nn.Linear(32, n_actions)
        # self.var_l.weight.data.mul_(0.1)

        self.logstd = nn.Parameter(torch.zeros(n_actions))

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
# class FeedForwardNN(nn.Module):
# 	"""
# 		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
# 	"""
# 	def __init__(self, in_dim, out_dim):
# 		"""
# 			Initialize the network and set up the layers.

# 			Parameters:
# 				in_dim - input dimensions as an int
# 				out_dim - output dimensions as an int

# 			Return:
# 				None
# 		"""
# 		super(FeedForwardNN, self).__init__()

# 		self.layer1 = nn.Linear(in_dim, 64)
# 		self.layer2 = nn.Linear(64, 64)
# 		self.layer3 = nn.Linear(64, out_dim)

# 	def forward(self, obs):
# 		"""
# 			Runs a forward pass on the neural network.

# 			Parameters:
# 				obs - observation to pass as input

# 			Return:
# 				output - the output of our forward pass
# 		"""
		# Convert observation to tensor if it's a numpy array
		# print(type(obs))
		# if isinstance(obs, tuple):
		# 	obs = torch.tensor(obs[0], dtype=torch.float)
		# if isinstance(obs,np.ndarray):
		# 	obs = torch.tensor(obs, dtype=torch.float)

# 		activation1 = F.relu(self.layer1(obs))
# 		activation2 = F.relu(self.layer2(activation1))
# 		output = self.layer3(activation2)

# 		return output
