import copy
import functools
import operator
import os
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.imitation.iil_dagger.model import Squeezenet
from learning.imitation.iil_dagger.utils.transform import transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class CriticCNN(nn.Module):
    def __init__(self, action_dim):
        super(CriticCNN, self).__init__()

        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 256)
        self.lin2 = nn.Linear(256 + action_dim, 128)
        self.lin3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = self.bn1(self.lr(self.conv1(states)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        try:
            x = x.view(x.size(0), -1)  # flatten
        except RuntimeError:
            x = x.reshape(x.size(0), -1)
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(torch.cat([x, actions], 1)))  # c
        x = self.lin3(x)

        return x

class DDPG(object):
    def __init__(self, policy, save_manager):
        super(DDPG, self).__init__()

        self.max_velocity = 0.7 # TODO make this a HP
        self.action_bounds = [self.max_velocity, 1]

        self.save_manager = save_manager

        self.base_policy = policy

        self.actor = Squeezenet(2, max_velocity=1).to(device)  # dont limit max vel to patch up to 8
        self.actor.final_conv.weight.data.fill_(0)
        self.critic = CriticCNN(2).to(device) # Squeezenet(1, 1, 1, is_critic=True).to(device)

        self.actor_target = copy.copy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)   # TODO lr as HP

        self.critic_target = CriticCNN(2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        torch.autograd.set_detect_anomaly(True)

    def clip(self, actions):    # Clip for steering
        # TODO the iil thing doens't clip, do we have to clip DDPG? Seems like no, because otherwise we can't cancel out
        # a -11 with a 1, you know?

        try:
            actions = torch.clamp(actions, 0, 1)
        except:
            actions[0] = np.clip(actions[0], 0, self.max_velocity)
            actions[1] = np.clip(actions[1], -1, 1)

        return actions

    def actor_act(self, actor, state): # TODO
        patch = actor(state)

        base = self.base_policy.mass_predict(state)

        combined = patch + base
        return combined

    def critic_crit(self, critic, states, actions): # TODO
        patch = critic(states)

        base = self.base_policy.mass_predict(states)

        combined = self.clip(patch + base)
        return combined

    def predict(self, state):
        state = transform([state], as_tensor=True, only_obs=True)

        patch = self.actor.predict(state).detach().cpu().numpy()
        base = self.base_policy.predict(state, already_transformed=True)

        return patch + base

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):
        time_for_trans = 0
        time_for_calc = 0

        for it in tqdm(range(iterations)):

            # Sample replay buffer
            temp = time.time()  # TODO

            state, next_state, action, reward, done = replay_buffer.sample(batch_size)

            time_for_trans += time.time() - temp    # TODO

            temp = time.time()  # TODO
            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_act(self.actor_target, next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor_act(self.actor, state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            time_for_calc += time.time() - temp  # TODO

        print(time_for_trans)
        print(time_for_calc)

    def save(self, filename):
        try:
            torch.save(self.actor.state_dict(), str(self.save_manager) + "/RPL/" + filename + "_actor.pth")
            torch.save(self.critic.state_dict(), str(self.save_manager) + "/RPL/" + filename + "_critic.pth")
        except:
            os.mkdir(os.getcwd() + "/" + str(self.save_manager) + "/RPL")
            torch.save(self.actor.state_dict(), str(self.save_manager) + "/RPL/" + filename + "_actor.pth")
            torch.save(self.critic.state_dict(), str(self.save_manager) + "/RPL/" + filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename+"model_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(filename+"model_critic.pth", map_location=device))
