import math

from gym.wrappers import FrameStack

from gym_duckietown.envs import DuckietownEnv
import argparse

from learning.imitation.iil_dagger.learner.DDPG import DDPG
from learning.imitation.iil_dagger.learning.iil_train_loop import train_iil
from .teacher import PurePursuitPolicy
from .learner import NeuralNetworkPolicy
from .model import Squeezenet
from .algorithms import DAgger
from .utils import MemoryMapDataset
import torch
import os

def launch_env(map_name, randomize_maps_on_reset=False, domain_rand=False, frame_stacking=1):
    environment = DuckietownEnv(
        domain_rand=domain_rand,
        max_steps=math.inf,
        map_name=map_name,
        randomize_maps_on_reset=randomize_maps_on_reset
    )

    tmp = environment._get_tile

    if frame_stacking != 1:
        environment = FrameStack(environment, frame_stacking)
        environment._get_tile = tmp
        environment.reset() # Force reset to get fake frame buffer

    return environment

def teacher(env, max_velocity):
    return PurePursuitPolicy(
        env=env,
        ref_velocity=max_velocity
    )

def legacy_parse(arg, arr):
    arg = str(arg)
    if "." in arg or "e" in arg:
        return float(arg)
    else:
        return arr[int(arg)]

def learning_rate_parse(arg):
    legacy_parse(arg, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

def decay_parse(arg):
    legacy_parse(arg, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])

# Best: Horizon64_LR1e-05_Decay0.95_BS32_epochs10_2020-06-0817:13:35.067047
def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', '-i', default=90, type=int)
    parser.add_argument('--horizon', '-r', default=64, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-3, type=learning_rate_parse)
    parser.add_argument('--decay', '-d', default=0.7, type=decay_parse)    # mixing decay
    parser.add_argument('--save-path', '-s', default='iil_baseline', type=str)
    parser.add_argument('--map-name', '-m', default="loop_empty", type=str)
    parser.add_argument('--num-outputs', '-n', default=2, type=int)

    parser.add_argument('--frame-stack', '-f', default=1, type=int) # TODO
    parser.add_argument('--static-mixing', default=False, type=bool)  # TODO
    return parser

if __name__ == '__main__':
    parser = process_args()
    input_shape = (120,160)
    batch_size = 32
    epochs = 10
    # Max velocity
    max_velocity = 0.7

    config = parser.parse_args()
    # check for  storage path
    if not(os.path.isdir(config.save_path)):
        os.makedirs(config.save_path)
    # launching environment
    environment = launch_env(config.map_name, randomize_maps_on_reset=True, frame_stacking=config.frame_stack)
    
    task_horizon = config.horizon
    task_episode = config.episode

    model = Squeezenet(num_outputs=config.num_outputs, max_velocity=max_velocity)

    #ddpg = DDPG(model)

    policy_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    dataset = MemoryMapDataset(25000, (3, *input_shape), (2,), config.save_path)
    learner = NeuralNetworkPolicy(
        model=model,
        optimizer=policy_optimizer,
        storage_location=config.save_path,
        graph_name=f"Horizon{config.horizon}_LR{config.learning_rate}_Decay{config.decay}_BS{batch_size}_epochs{epochs}_static-mixing{config.static_mixing}_",
        batch_size=batch_size,
        epochs=epochs,
        input_shape=input_shape,
        max_velocity = max_velocity,
        dataset = dataset
    )

    
    algorithm = DAgger(env=environment,
                        teacher=teacher(environment, max_velocity),
                        learner=learner,
                        horizon = task_horizon,
                        episodes=task_episode,
                        alpha = config.decay,
                       )
    
    algorithm.train(debug=True)  #DEBUG to show simulation

    #train_iil(environment, learner, task_horizon, task_episode, config.decay, max_velocity=0.7, _debug=False)

    print("Finished training. Closing the env now...")
    environment.close()
    print("Closed successfully.")
    exit()


