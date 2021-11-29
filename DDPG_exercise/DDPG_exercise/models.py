import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable
import numpy as np

class Critic(nn.Module):
    def __init__(self, input_size_S, Layer_S, output_size_S, input_size_A, Layer_A, output_size_A):
        super(Critic, self).__init__()
        self.linear1S = nn.Linear(input_size_S, Layer_S)
        self.linear2S = nn.Linear(Layer_S, output_size_S)

        self.output_size_S = output_size_S

        self.linear1A = nn.Linear(input_size_A, Layer_A)

        #self.linearHybrid=nn.Linear(400, 200)
        self.linearHybrid=nn.Linear(output_size_S+Layer_A,200)
        self.final=nn.Linear(200,1)

        self.Layer_A = Layer_A
    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = F.relu(self.linear1S(state))
        x = F.linear(self.linear2S(x), torch.FloatTensor(np.eye(self.output_size_S)))

        y = F.linear(self.linear1A(action), torch.FloatTensor(np.eye(self.Layer_A)))

        z = torch.cat([x, y], 1)
        z = F.relu(self.linearHybrid(z))
        z = F.linear(self.final(z), torch.FloatTensor([1]))

        return z

class Actor(nn.Module):
    def __init__(self, input_size, Layer_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, Layer_size[0])
        self.linear2 = nn.Linear(Layer_size[0], Layer_size[1])
        self.linear3 = nn.Linear(Layer_size[1], output_size)
        
    def forward(self, state, P_max_ref):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = F.linear(x, P_max_ref, bias=None)


        return x

