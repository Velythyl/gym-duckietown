import copy

from learning.imitation.iil_dagger.learner.DDPG import DDPG
from learning.imitation.iil_dagger.teacher import PurePursuitPolicy
from .train import launch_env, teacher
from .learner import NeuralNetworkPolicy
from .model import Squeezenet
import argparse
import os
import numpy as np
import torch
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', '-c', default=True, type=str2bool)

    parser.add_argument('--episode', '-i', default=256, type=int)
    parser.add_argument('--save-path', '-s', default='iil_baseline', type=str)
    parser.add_argument('--model-path', '-mp', default="iil_baseline", type=str)
    parser.add_argument('--model-name', default="model.pt", type=str)
    parser.add_argument('--map-name', '-m', default="loop_empty", type=str)

    parser.add_argument('--RPL_path', default="RPL", type=str)
    return parser

if __name__ == '__main__':
    parser = process_args()
    input_shape = (120,160)
    max_velocity = 0.7

    config = parser.parse_args()
    # launching environment and testing on different maps using map randomization
    environment = launch_env(config.map_name, randomize_maps_on_reset=False)

    task_episode = config.episode

    model = Squeezenet(num_outputs=2, max_velocity=max_velocity)

    policy = NeuralNetworkPolicy(
        model=model,
        optimizer= None,
        dataset=None,
        storage_location="",
        input_shape=input_shape,
        max_velocity = max_velocity,
        model_path = config.model_path+"/"+config.model_name
    )

    if False:
        try:
            rpl = DDPG(policy, None, None)
            rpl.load(config.model_path+"/"+config.RPL_path+"/") # TODO make this prettier
            policy = rpl
        except Exception as e:
            print(e)
            print("Could not load RPL: is it missing? Assuming this is intended; continuing with Dagger")
            pass

    GAIN = 10

    with torch.no_grad():
        while True:
            obs = environment.reset()
            environment.render()
            rewards = []

            nb_of_steps = 0

            while True:
                action = list(policy.predict(np.array(obs)))

                obs, rew, done, misc = environment.step(action)
                rewards.append(rew)
                environment.render()

                nb_of_steps += 1

                if done or nb_of_steps > config.episode:
                    break
            print("mean episode reward:", np.mean(rewards))

    environment.close()



