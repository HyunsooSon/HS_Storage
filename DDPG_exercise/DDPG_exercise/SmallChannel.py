import torch
import math
import random


class SmallGenerator():
    def __init__(self, dim_size, rho):
        self.dim_size = dim_size
        self.rho = rho

    def First_channel(self):
        smallChannel=1/math.sqrt(2)*(torch.randn(self.dim_size, self.dim_size)+1j*torch.randn(self.dim_size, self.dim_size))
        return smallChannel

    def channel_Evol(self, smallChannel):
        smallChannel_evol = 1/math.sqrt(2)*(torch.randn(self.dim_size, self.dim_size)+1j*torch.randn(self.dim_size, self.dim_size))
        smallChannel= self.rho*smallChannel + (math.sqrt(1-self.rho**2))*smallChannel_evol
        return smallChannel