import torch
import random
import numpy as np

class Environ():
    def __init__(self, BS_Position, min_D, max_D, exponent):
        self.BS_Position = BS_Position
        self.min_D = min_D
        self.max_D = max_D
        self.exponent = exponent
        self.BS_num = len(self.BS_Position)
        self.UE_num = self.BS_num

    def CreateNew(self):
        UE_Position = torch.zeros((self.BS_num, 2))
        for UEidx in range(self.BS_num):
            cdf_position = torch.rand(1)
            UE_distance = torch.sqrt(cdf_position*(self.max_D[UEidx]**2)+(1-cdf_position)*(self.min_D[UEidx]**2))
            UE_angle = torch.rand(1) * 2 * np.pi
            UE_Position[UEidx] = torch.Tensor([UE_distance * torch.cos(UE_angle), UE_distance * torch.sin(UE_angle)])

        UE_Position = UE_Position + self.BS_Position
        Large_cahnnel = torch.zeros((self.BS_num, self.UE_num))

        for i in range(self.BS_num):
            for j in range(self.UE_num):
                Large_cahnnel[i][j] = (10**(12.09))*(np.linalg.norm(self.BS_Position[i] - UE_Position[j]) ** self.exponent)

        return UE_Position, Large_cahnnel






