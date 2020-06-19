import copy
import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.imitation.iil_dagger.model import Squeezenet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class DDPG(object):
    def __init__(self, policy):
        super(DDPG, self).__init__()

        self.policy = policy

        self.actor = Squeezenet(2, max_velocity=1)  # dont limit max vel to patch up to 8
        self.critic = Squeezenet(1, 1, 1, is_critic=True)

        self.actor_target = copy.copy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)   # TODO lr as HP

        self.critic_target = copy.copy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def actor_act(self, state):
        state_actor = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state_actor) + self.policy.predict(state)

    def predict(self, state):
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3
        return self.actor_act(state).detach().cpu().numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=False)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
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
            actor_loss = -self.critic(state, self.actor_act(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        torch.save(self.critic.state_dict(), '{}/{}_critic.pth'.format(directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict()
        self.critic.load_state_dict()
