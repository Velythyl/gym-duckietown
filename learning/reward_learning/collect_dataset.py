import math
import os
import random
import time

import cv2
import numpy as np

from gym_duckietown.envs import DuckietownEnv
import math
import numpy as np
import math
import numpy as np
from gym_duckietown.simulator import AGENT_SAFETY_RAD
from learning.reward_learning.agents.good_ppc import PurePursuitPolicy
from learning.reward_learning.utils.isolate_car import isolate_car


def seed(seed):
    #torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
seed(random.randint(0, 9999999))

os.chdir("../../src/gym_duckietown")
print(os.listdir())

environment = DuckietownEnv(
        domain_rand=False,
        max_steps=math.inf,
        randomize_maps_on_reset=False,
        map_name="loop_empty"
    )

policy = PurePursuitPolicy(environment)

NB_OF_OBS = 60000

dataset_obs = np.zeros((NB_OF_OBS, 32, 32, 3), dtype="uint8")
dataset_rew = np.zeros(NB_OF_OBS, dtype=float)
dataset_index = 0

MAX_STEPS = 1000

start = time.time()

while dataset_index < NB_OF_OBS:
    nb_of_steps = 0
    for nb_of_steps in range(MAX_STEPS):
        action = list(policy.predict(None))
        action[1]*=7

        obs, rew, done, _ = environment.step(action, True, True)
        if done or dataset_index >= NB_OF_OBS:
            break

        dataset_obs[dataset_index] = cv2.resize(isolate_car(obs, environment), (32, 32))
        dataset_rew[dataset_index] = rew
        dataset_index += 1
        print(dataset_index)

        nb_of_steps += 1
environment.close()

print("TOOK", time.time()-start)

np.save("../../learning/reward_learning/dataset/obs.npy", dataset_obs)
np.save("../../learning/reward_learning/dataset/rew.npy", dataset_rew)