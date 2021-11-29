import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import torch
from EnvironmentSetting import Environ
from SmallChannel import SmallGenerator as SmallG
from models import *
from ddpg import *
import pandas



# Environment setting
AP_Position=torch.tensor([[0, 0], [0.5, 0], [0, 0.5], [-0.5, 0], [0, -0.5]])  # in km
AP_num = len(AP_Position)
min_D=[0.01, 0.01, 0.01, 0.01, 0.01]
max_D=[1, 0.2, 0.2, 0.2, 0.2]

noise_var = 10**(-11.4) # Noise variance -114dBm
rate_list=[]

# Propagation Property
path_loss_exp = 3.76
[UE_Position, Large_Scale] = Environ(AP_Position, min_D, max_D, path_loss_exp).CreateNew()
print(UE_Position)
LogNorm_std = 10**(0.8) # log_normal shadowing std = 8 dB
Shadow_Eff = torch.exp(-LogNorm_std *torch.randn(AP_num, AP_num))
rho = 1

# Power constraint
P1_max = 10**3
P2_max = 10**(2.3)
P3_max = 10**(2.3)
P4_max = 10**(2.3)
P5_max = 10**(2.3)

#  batch size
batch_size = 16

Central_DDPG = DDPGagent()

BW = 10**7
reward=[]
reward_random=[]
reward_full=[]

avg_reward=[]

for step in range(2000):
    if step == 0:
        SmallChannel_first = SmallG(AP_num, rho).First_channel()  # Generating the first channel
        System_Channel = 1/torch.sqrt(Large_Scale)*torch.sqrt(Shadow_Eff)*SmallChannel_first   # AP x UE
        Desired_Channel_Gain = torch.abs(torch.eye(AP_num)*System_Channel)**2
        Interfer_Channel_Gain = torch.abs((torch.ones((AP_num, AP_num))-torch.eye(AP_num))*System_Channel)**2

        UE1_states = torch.FloatTensor([0, 0, 0, 0, 0, Desired_Channel_Gain[0][0], 0])
        UE2_states = torch.FloatTensor([0, 0, 0, 0, 0, Desired_Channel_Gain[1][1], 0])
        UE3_states = torch.FloatTensor([0, 0, 0, 0, 0, Desired_Channel_Gain[2][2], 0])
        UE4_states = torch.FloatTensor([0, 0, 0, 0, 0, Desired_Channel_Gain[3][3], 0])
        UE5_states = torch.FloatTensor([0, 0, 0, 0, 0, Desired_Channel_Gain[4][4], 0])
        Universal_states = (torch.abs(System_Channel)**2).view(-1)


        UE1_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P1_max, P1_max))
        UE2_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P2_max, P2_max))
        UE3_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P3_max, P3_max))
        UE4_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P4_max, P4_max))
        UE5_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P5_max, P5_max))

        Action_random = torch.FloatTensor([UE1_action_random, UE2_action_random, UE3_action_random, UE4_action_random, UE5_action_random])
        num_random = torch.matmul(Action_random, Desired_Channel_Gain)
        deno_random = torch.matmul(Action_random, Interfer_Channel_Gain) + noise_var
        SINR_random = num_random / deno_random

        Rate_random = torch.log2(torch.ones((1, AP_num)) + SINR_random)
        reward_new_random = torch.mean(Rate_random)
        reward_random.append(reward_new_random)


        Action_full = torch.FloatTensor([P1_max, P2_max, P3_max, P4_max, P5_max])
        num_full = torch.matmul(Action_full, Desired_Channel_Gain)
        deno_full = torch.matmul(Action_full, Interfer_Channel_Gain) + noise_var
        SINR_full = num_full / deno_full

        Rate_full = torch.log2(torch.ones((1, AP_num)) + SINR_full)
        reward_new_full = torch.mean(Rate_full)
        reward_full.append(reward_new_full)


        UE1_action = torch.FloatTensor(np.minimum(np.random.rand(1)*P1_max, P1_max))
        UE2_action = torch.FloatTensor(np.minimum(np.random.rand(1)*P2_max, P2_max))
        UE3_action = torch.FloatTensor(np.minimum(np.random.rand(1)*P3_max, P3_max))
        UE4_action = torch.FloatTensor(np.minimum(np.random.rand(1)*P4_max, P4_max))
        UE5_action = torch.FloatTensor(np.minimum(np.random.rand(1)*P5_max, P5_max))

        Action = torch.FloatTensor([UE1_action, UE2_action, UE3_action, UE4_action, UE5_action])
        num = torch.matmul(Action, Desired_Channel_Gain)
        deno = torch.matmul(Action, Interfer_Channel_Gain) + noise_var
        SINR = num / deno

        Rate = torch.log2(torch.ones((1, AP_num)) + SINR)
        reward_new = torch.mean(Rate)
        reward.append(reward_new)

        System_Channel_pre = System_Channel
        Desired_Channel_Gain_pre = Desired_Channel_Gain
        Interfer_Channel_Gain_pre = Interfer_Channel_Gain
        Action_pre = Action
        Desired_pre = num
        Interfere_pre = torch.matmul(Action, Interfer_Channel_Gain)
        SINR_pre = SINR

        UE1_states_pre = UE1_states
        UE2_states_pre = UE2_states
        UE3_states_pre = UE3_states
        UE4_states_pre = UE4_states
        UE5_states_pre = UE5_states
        Universal_states_pre = Universal_states

    else:
        SmallChannel = SmallG(AP_num, rho).channel_Evol(SmallChannel_first) # Channel variation
        System_Channel = 1/torch.sqrt(Large_Scale)*torch.sqrt(Shadow_Eff)*SmallChannel   # AP x UE
        Desired_Channel_Gain = torch.abs(torch.eye(AP_num)*System_Channel)**2
        Interfer_Channel_Gain = torch.abs((torch.ones((AP_num, AP_num))-torch.eye(AP_num))*System_Channel)**2
        Interfere_prediction = torch.matmul(Action_pre, Interfer_Channel_Gain)

        UE1_states = torch.FloatTensor([Desired_Channel_Gain_pre[0][0], Action_pre[0], Interfere_pre[0], SINR_pre[0], Rate[0][0], Desired_Channel_Gain[0][0], Interfere_prediction[0]])
        UE2_states = torch.FloatTensor([Desired_Channel_Gain_pre[1][1], Action_pre[1], Interfere_pre[1], SINR_pre[1], Rate[0][1], Desired_Channel_Gain[1][1], Interfere_prediction[1]])
        UE3_states = torch.FloatTensor([Desired_Channel_Gain_pre[2][2], Action_pre[2], Interfere_pre[2], SINR_pre[2], Rate[0][2], Desired_Channel_Gain[2][2], Interfere_prediction[2]])
        UE4_states = torch.FloatTensor([Desired_Channel_Gain_pre[3][3], Action_pre[3], Interfere_pre[3], SINR_pre[3], Rate[0][3], Desired_Channel_Gain[3][3], Interfere_prediction[3]])
        UE5_states = torch.FloatTensor([Desired_Channel_Gain_pre[4][4], Action_pre[4], Interfere_pre[4], SINR_pre[4], Rate[0][4], Desired_Channel_Gain[4][4], Interfere_prediction[4]])
        Universal_states = torch.FloatTensor((torch.abs(System_Channel)**2).reshape(-1))

        System_Channel_pre_norm = torch.abs(System_Channel_pre)**2
        System_Channel_norm = torch.abs(System_Channel) ** 2


        memory_state_pre = torch.cat((UE1_states_pre, UE2_states_pre, UE3_states_pre, UE4_states_pre, UE5_states_pre, Universal_states_pre),0)
        memory_action_pre = Action_pre
        memory_reward = reward_new
        memory_state = torch.cat((UE1_states, UE2_states, UE3_states, UE4_states, UE5_states, Universal_states),0)

        Central_DDPG.memory.push(memory_state_pre, memory_action_pre, memory_reward, memory_state)

        UE1_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P1_max, P1_max))
        UE2_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P2_max, P2_max))
        UE3_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P3_max, P3_max))
        UE4_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P4_max, P4_max))
        UE5_action_random = torch.FloatTensor(np.minimum(np.random.rand(1)*P5_max, P5_max))

        Action_random = torch.FloatTensor([UE1_action_random, UE2_action_random, UE3_action_random, UE4_action_random, UE5_action_random])
        num_random = torch.matmul(Action_random, Desired_Channel_Gain)
        deno_random = torch.matmul(Action_random, Interfer_Channel_Gain) + noise_var
        SINR_random = num_random / deno_random

        Rate_random = torch.log2(torch.ones((1, AP_num)) + SINR_random)
        reward_new_random = torch.mean(Rate_random)
        reward_random.append(reward_new_random)

        Action_full = torch.FloatTensor([P1_max, P2_max, P3_max, P4_max, P5_max])
        num_full = torch.matmul(Action_full, Desired_Channel_Gain)
        deno_full = torch.matmul(Action_full, Interfer_Channel_Gain) + noise_var
        SINR_full = num_full / deno_full

        Rate_full = torch.log2(torch.ones((1, AP_num)) + SINR_full)
        reward_new_full = torch.mean(Rate_full)
        reward_full.append(reward_new_full)


        Action = Central_DDPG.get_action(System_Channel_pre_norm, System_Channel_norm, Action_pre)


        num = torch.matmul(Action, Desired_Channel_Gain)
        deno = torch.matmul(Action, Interfer_Channel_Gain) + noise_var
        SINR = num / deno

        Rate = torch.log2(torch.ones((1, AP_num)) + SINR)

        reward_new = torch.mean(Rate)
        reward.append(reward_new)

        if len(Central_DDPG.memory) > batch_size:
            buffer_states_sample, buffer_actions_sample, buffer_rewards_sample, buffer_next_states_sample = Central_DDPG.memory.sample(batch_size)
            Central_DDPG.update(batch_size)

            if step % 100 == 0:
                Central_DDPG.target_update()



        System_Channel_pre = System_Channel
        Desired_Channel_Gain_pre = Desired_Channel_Gain
        Interfer_Channel_Gain_pre = Interfer_Channel_Gain
        Action_pre = Action
        Desired_pre = num
        Interfere_pre = torch.matmul(Action, Interfer_Channel_Gain)
        SINR_pre = SINR

        UE1_states_pre = UE1_states
        UE2_states_pre = UE2_states
        UE3_states_pre = UE3_states
        UE4_states_pre = UE4_states
        UE5_states_pre = UE5_states
        Universal_states_pre = Universal_states

        if step % 100 == 0:
            print(step)
            print(np.mean(reward_random))
            print(np.mean(reward_full))
            print(np.mean(reward))

reward_random_MVE=[]
reward_full_MVE=[]
reward_MVE=[]
for i in range(step):
    if i >= 200:
        reward_random_MVE.append(np.mean(reward_random[i-200:i+1]))
        reward_full_MVE.append(np.mean(reward_full[i-200:i+1]))
        reward_MVE.append(np.mean(reward[i-200:i+1]))



plt.figure(0)
plt.plot(reward_random_MVE, 'r--', linewidth=1)
plt.plot(reward_full_MVE, 'g:', linewidth=1)
plt.plot(reward_MVE, 'b-', linewidth=1)
plt.grid(True)

plt.show()
#print(SmallChannel)


