import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *
from utils import *


class DDPGagent:
    def __init__(self, N=5, actor_learning_rate=1e-3, critic_learning_rate=1e-3, gamma=0.5, tau=1e-4, max_memory_size=1000):
        # Params
        self.gamma = gamma
        self.tau = tau
        self.N   = N
        # Networks
        Actor_first = 7
        Actor_mid = [100, 100]
        Actor_final = 1

        self.UE1_Actor = Actor(Actor_first, Actor_mid, Actor_final)
        self.UE1_Actor_target = Actor(Actor_first, Actor_mid, Actor_final)

        self.UE2_Actor = Actor(Actor_first, Actor_mid, Actor_final)
        self.UE2_Actor_target = Actor(Actor_first, Actor_mid, Actor_final)

        self.UE3_Actor = Actor(Actor_first, Actor_mid, Actor_final)
        self.UE3_Actor_target = Actor(Actor_first, Actor_mid, Actor_final)

        self.UE4_Actor = Actor(Actor_first, Actor_mid, Actor_final)
        self.UE4_Actor_target = Actor(Actor_first, Actor_mid, Actor_final)

        self.UE5_Actor = Actor(Actor_first, Actor_mid, Actor_final)
        self.UE5_Actor_target = Actor(Actor_first, Actor_mid, Actor_final)

        input_size_S   = self.N**2+self.N*7
        Layer_S        = 200
        output_size_S  = 200
        input_size_A   = self.N
        Layer_A        = 200
        output_size_A  = 1


        self.critic        = Critic(input_size_S, Layer_S, output_size_S, input_size_A, Layer_A, output_size_A)
        self.critic_target = Critic(input_size_S, Layer_S, output_size_S, input_size_A, Layer_A, output_size_A)

        for target_param_UE1, param_UE1 in zip(self.UE1_Actor_target.parameters(), self.UE1_Actor.parameters()):
            target_param_UE1.data.copy_(tau * param_UE1.data + (1.0 - tau) * target_param_UE1.data)

        for target_param_UE2, param_UE2 in zip(self.UE2_Actor_target.parameters(), self.UE2_Actor.parameters()):
            target_param_UE2.data.copy_(tau * param_UE2.data + (1.0 - tau) * target_param_UE2.data)

        for target_param_UE3, param_UE3 in zip(self.UE3_Actor_target.parameters(), self.UE3_Actor.parameters()):
            target_param_UE3.data.copy_(tau * param_UE3.data + (1.0 - tau) * target_param_UE3.data)

        for target_param_UE4, param_UE4 in zip(self.UE4_Actor_target.parameters(), self.UE4_Actor.parameters()):
            target_param_UE4.data.copy_(tau * param_UE4.data + (1.0 - tau) * target_param_UE4.data)

        for target_param_UE5, param_UE5 in zip(self.UE5_Actor_target.parameters(), self.UE5_Actor.parameters()):
            target_param_UE5.data.copy_(tau * param_UE5.data + (1.0 - tau) * target_param_UE5.data)


        for target_param_Cr, param_Cr in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param_Cr.data.copy_(tau * param_Cr.data + (1.0 - tau) * target_param_Cr.data)

        # Training

        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()

        self.UE1_Actor_optimizer = optim.Adam(self.UE1_Actor.parameters(), lr=actor_learning_rate)
        self.UE2_Actor_optimizer = optim.Adam(self.UE2_Actor.parameters(), lr=actor_learning_rate)
        self.UE3_Actor_optimizer = optim.Adam(self.UE3_Actor.parameters(), lr=actor_learning_rate)
        self.UE4_Actor_optimizer = optim.Adam(self.UE4_Actor.parameters(), lr=actor_learning_rate)
        self.UE5_Actor_optimizer = optim.Adam(self.UE5_Actor.parameters(), lr=actor_learning_rate)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, System_Channel_pre_norm, System_Channel_norm, actions_pre):
        self.System_Channel_pre_norm = System_Channel_pre_norm
        self.System_Channel_norm = System_Channel_norm
        self.actions_pre= actions_pre

        Desired_Channel_Gain_pre = torch.FloatTensor(torch.eye(self.N) * self.System_Channel_pre_norm)
        Interfer_Channel_Gain_pre = torch.FloatTensor((torch.ones((self.N, self.N)) - torch.eye(self.N)) * self. System_Channel_pre_norm)
        Interferences             = torch.FloatTensor(torch.matmul(self.actions_pre, (Interfer_Channel_Gain_pre)))
        Interfer_Channel_Gain_now = torch.FloatTensor((torch.ones((self.N, self.N)) - torch.eye(self.N)) * self. System_Channel_norm)
        Interferences_predict     = torch.FloatTensor(torch.matmul(self.actions_pre, (Interfer_Channel_Gain_now)))

        num_pre = torch.matmul(self.actions_pre, Desired_Channel_Gain_pre)
        noise_var = 10 ** (-11.4)  # Noise variance -114dBm
        deno_pre = Interferences + noise_var
        SINR_pre = torch.ones((1,self.N))+num_pre/deno_pre
        rate_pre = np.log2(1+SINR_pre)
        P_max = torch.FloatTensor([[10**3], [10**2.3], [10**2.3], [10**2.3], [10**2.3]])

        UE_1_state  = torch.FloatTensor([Desired_Channel_Gain_pre[0][0], self.actions_pre[0], Interferences[0], SINR_pre[0][0], rate_pre[0][0], Interfer_Channel_Gain_now[0][0], Interferences_predict[0] ])
        UE_1_action = self.UE1_Actor.forward(UE_1_state, P_max[0])
        UE_1_action = torch.maximum(UE_1_action+np.sqrt(2)*torch.randn(1),torch.tensor([0]))
        UE_1_action = torch.minimum(UE_1_action, P_max[0])
        UE_1_action = UE_1_action.detach()

        UE_2_state  = torch.FloatTensor([Desired_Channel_Gain_pre[1][1], self.actions_pre[1], Interferences[1], SINR_pre[0][1], rate_pre[0][1], Interfer_Channel_Gain_now[1][1], Interferences_predict[1] ])
        UE_2_action = self.UE2_Actor.forward(UE_2_state, P_max[1])
        UE_2_action = torch.maximum(UE_2_action+np.sqrt(2)*torch.randn(1),torch.tensor([0]))
        UE_2_action = torch.minimum(UE_2_action, P_max[1])
        UE_2_action = UE_2_action.detach()

        UE_3_state  = torch.FloatTensor([Desired_Channel_Gain_pre[2][2], self.actions_pre[2], Interferences[2], SINR_pre[0][2], rate_pre[0][2], Interfer_Channel_Gain_now[2][2], Interferences_predict[2] ])
        UE_3_action = self.UE3_Actor.forward(UE_3_state, P_max[2])
        UE_3_action = torch.maximum(UE_3_action+np.sqrt(2)*torch.randn(1),torch.tensor([0]))
        UE_3_action = torch.minimum(UE_3_action, P_max[2])
        UE_3_action = UE_3_action.detach()

        UE_4_state  = torch.FloatTensor([Desired_Channel_Gain_pre[3][3], self.actions_pre[3], Interferences[3], SINR_pre[0][3], rate_pre[0][3], Interfer_Channel_Gain_now[3][3], Interferences_predict[3] ])
        UE_4_action = self.UE4_Actor.forward(UE_4_state, P_max[3])
        UE_4_action = torch.maximum(UE_4_action+np.sqrt(2)*torch.randn(1),torch.tensor([0]))
        UE_4_action = torch.minimum(UE_4_action, P_max[3])
        UE_4_action = UE_4_action.detach()

        UE_5_state  = torch.FloatTensor([Desired_Channel_Gain_pre[4][4], self.actions_pre[4], Interferences[4], SINR_pre[0][4], rate_pre[0][4], Interfer_Channel_Gain_now[4][4], Interferences_predict[4] ])
        UE_5_action = self.UE5_Actor.forward(UE_5_state, P_max[4])
        UE_5_action = torch.maximum(UE_5_action+np.sqrt(2)*torch.randn(1),torch.tensor([0]))
        UE_5_action = torch.minimum(UE_5_action, P_max[4])
        UE_5_action = UE_5_action.detach()

        return torch.Tensor([UE_1_action, UE_2_action, UE_3_action, UE_4_action, UE_5_action])

    def update(self, batch_size):
        buffer_states_sample, buffer_actions_sample, buffer_rewards_sample, buffer_next_states_sample = self.memory.sample(batch_size)



        buffer_states = torch.vstack(buffer_states_sample)
        buffer_actions = torch.vstack(buffer_actions_sample)
        buffer_rewards = torch.vstack(buffer_rewards_sample)
        buffer_next_states = torch.vstack(buffer_next_states_sample)

        buffer_next_UE1 = buffer_next_states[:, 0:7]
        buffer_next_UE2 = buffer_next_states[:, 7:14]
        buffer_next_UE3 = buffer_next_states[:, 14:21]
        buffer_next_UE4 = buffer_next_states[:, 21:28]
        buffer_next_UE5 = buffer_next_states[:, 28:35]


        P_max = torch.FloatTensor([[10 ** 3], [10 ** 2.3], [10 ** 2.3], [10 ** 2.3], [10 ** 2.3]])
        # Critic loss        
        Qvals =  self.critic.forward(buffer_states, buffer_actions).view([-1,1])
        UE1_target_out = self.UE1_Actor_target.forward(buffer_next_UE1, P_max[0])
        UE2_target_out = self.UE2_Actor_target.forward(buffer_next_UE2, P_max[1])
        UE3_target_out = self.UE3_Actor_target.forward(buffer_next_UE3, P_max[2])
        UE4_target_out = self.UE4_Actor_target.forward(buffer_next_UE4, P_max[3])
        UE5_target_out = self.UE5_Actor_target.forward(buffer_next_UE5, P_max[4])

        buffer_next_actions = torch.t(torch.reshape(torch.cat([UE1_target_out, UE2_target_out, UE3_target_out, UE4_target_out, UE5_target_out]), (self.N, len(UE1_target_out))))
        next_Q = self.critic_target.forward(buffer_next_states, buffer_next_actions.detach())
        Qprime = buffer_rewards + self.gamma * next_Q.view([-1,1])
        critic_loss = self.critic_criterion(Qvals, Qprime)
        # Actor loss

        buffer_UE1 = buffer_states[:, 0:7]
        buffer_UE2 = buffer_states[:, 7:14]
        buffer_UE3 = buffer_states[:, 14:21]
        buffer_UE4 = buffer_states[:, 21:28]
        buffer_UE5 = buffer_states[:, 28:35]

        UE1_out = self.UE1_Actor.forward(buffer_UE1, P_max[0])
        UE2_out = self.UE2_Actor.forward(buffer_UE2, P_max[1])
        UE3_out = self.UE3_Actor.forward(buffer_UE3, P_max[2])
        UE4_out = self.UE4_Actor.forward(buffer_UE4, P_max[3])
        UE5_out = self.UE5_Actor.forward(buffer_UE5, P_max[4])

        Actors_actions = torch.t(torch.reshape(torch.cat([UE1_out, UE2_out, UE3_out, UE4_out, UE5_out]), (self.N, len(UE1_out))))
        policy_loss = -self.critic.forward(buffer_states,  Actors_actions.detach()).mean()
        # update networks
        self.UE1_Actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.UE1_Actor_optimizer.step()

        self.UE2_Actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.UE2_Actor_optimizer.step()

        self.UE3_Actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.UE3_Actor_optimizer.step()

        self.UE4_Actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.UE4_Actor_optimizer.step()

        self.UE5_Actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.UE5_Actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

    def target_update(self):
        for target_param_UE1, param_UE1 in zip(self.UE1_Actor_target.parameters(), self.UE1_Actor.parameters()):
            target_param_UE1.data.copy_(param_UE1.data * self.tau + target_param_UE1.data * (1.0 - self.tau))

        for target_param_UE2, param_UE2 in zip(self.UE2_Actor_target.parameters(), self.UE2_Actor.parameters()):
            target_param_UE2.data.copy_(param_UE2.data * self.tau + target_param_UE2.data * (1.0 - self.tau))

        for target_param_UE3, param_UE3 in zip(self.UE3_Actor_target.parameters(), self.UE3_Actor.parameters()):
            target_param_UE3.data.copy_(param_UE3.data * self.tau + target_param_UE3.data * (1.0 - self.tau))

        for target_param_UE4, param_UE4 in zip(self.UE4_Actor_target.parameters(), self.UE4_Actor.parameters()):
            target_param_UE4.data.copy_(param_UE4.data * self.tau + target_param_UE4.data * (1.0 - self.tau))

        for target_param_UE5, param_UE5 in zip(self.UE5_Actor_target.parameters(), self.UE5_Actor.parameters()):
            target_param_UE5.data.copy_(param_UE5.data * self.tau + target_param_UE5.data * (1.0 - self.tau))

        for target_param_Cr, param_Cr in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param_Cr.data.copy_(param_Cr.data * self.tau + target_param_Cr.data * (1.0 - self.tau))

