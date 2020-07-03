import numpy as np

from learning.imitation.iil_dagger.learner.DDPG import DDPG
from learning.imitation.iil_dagger.teacher import PurePursuitPolicy
import pickle
import random
import resource
import gym_duckietown
import numpy as np
import torch
import gym
import os

from learning.imitation.iil_dagger.utils.transform import transform

"""
    parser.add_argument("--log_file", default=None, type=str)  # Maximum number of steps to keep in the replay buffer
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=250):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward

class ReplayBuffer(object):
    def __init__(self, max_size=1e6, shape=(120, 160)):
        self.states = torch.empty(size=(max_size, 3, shape[0], shape[1]))
        self.next_states = torch.empty(size=(max_size, 3, shape[0], shape[1]))
        self.actions = torch.empty(size=(max_size, 2))
        self.rewards = torch.empty(size=(max_size, 1))
        self.dones = torch.empty(size=(max_size, 1))

        self._count = 0
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        if self._count < self.max_size:
            index = self._count

            self._count += 1
        else:
            # Remove random element in the memory before adding a new one
            index = random.randrange(self.max_size)

        self.states[index] = transform([state], as_tensor=True, on_device=False, only_obs=True)[0]
        self.next_states[index] = transform([next_state], as_tensor=True, on_device=False, only_obs=True)[0]
        self.actions[index] = torch.from_numpy(action)
        self.rewards[index] = reward
        self.dones[index] = done


    def sample(self, batch_size=100):
        ind = np.random.randint(0, self._count, size=batch_size)

        state = self.states[ind]
        next_state = self.next_states[ind]
        action = self.actions[ind]
        reward = self.rewards[ind]
        done = self.dones[ind]

        return state.to(device), next_state.to(device), action.to(device), reward.to(device), done.to(device)

        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            states.append(np.array(state, copy=False))
            next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1,1),
            "done": np.stack(dones).reshape(-1,1)
        }


def train_rpl(
        env,
        rpl_policy,
        start_timesteps,
        eval_freq,
        max_timesteps,
        expl_noise,
        batch_size,
        discount,
        tau,
        env_timesteps,
        replay_buffer_max_size
    ):

    """
    if args.log_file != None:
        print('You asked for a log file. "Tee-ing" print to also print to file "' + args.log_file + '" now...')

        import subprocess, os, sys

        tee = subprocess.Popen(["tee", args.log_file], stdin=subprocess.PIPE)
        # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
        # of any child processes we spawn)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    """

    env.reset()

    replay_buffer = ReplayBuffer(replay_buffer_max_size)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, rpl_policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0

    while total_timesteps < max_timesteps:

        if done:

            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                rpl_policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, warming_up=total_timesteps < start_timesteps)

            # Evaluate episode
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                evaluations.append(evaluate_policy(env, rpl_policy))
                rpl_policy.save()
                print("Saving model")
                #np.savez(rpl_policy.), evaluations)

            # Reset environment
            env_counter += 1
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < start_timesteps:
            action = env.action_space.sample()
        else:
            action = rpl_policy.predict(np.array(obs))
            if expl_noise != 0:
                action = (action + np.random.normal(
                    0,
                    expl_noise,
                    size=env.action_space.shape[0])
                          ).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        if action[
            0] < 0.001:  # Penalise slow actions: helps the bot to figure out that going straight > turning in circles
            reward = 0

        if episode_timesteps >= env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(env, rpl_policy))

    rpl_policy.save()
    print("Saving model")
