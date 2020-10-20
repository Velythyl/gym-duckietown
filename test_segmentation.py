import math
import os
import random

import numpy as np

# Question 1: investigation:
"""
import cv2

img = cv2.imread("./PennFudanPed/PNGImages/FudanPed00001.png", cv2.IMREAD_COLOR)
cv2.imshow("image", img)
cv2.waitKey(0)

mask = cv2.imread("./PennFudanPed/PedMasks/FudanPed00001_mask.png", cv2.IMREAD_UNCHANGED)
mask = np.floor(mask / np.max(mask) * 255).astype(np.uint8)
cv2.imshow("image", cv2.applyColorMap(mask, cv2.COLORMAP_RAINBOW))
cv2.waitKey(0)
"""

# Question 2: data gathering

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import AGENT_SAFETY_RAD



def seed(seed):
    # torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed(random.randint(0, 9999999))

from PIL import Image

def to_image(np_array):

    img = Image.fromarray(np_array, 'RGB')
    img.show()
    i = 0


os.chdir("./src/gym_duckietown")

environment = DuckietownEnv(
    domain_rand=False,
    max_steps=math.inf,
    randomize_maps_on_reset=False,
    map_name="udem_spooky"
)

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = list(policy.predict(np.array(obs)))
        action[1] *= 7

        obs, rew, done, misc = environment.step(action)
        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        # to_image(obs)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
    print("mean episode reward:", np.mean(rewards))

environment.close()


# Question 3: data annotation & cleaning

# todo: remove all non-duckies from the masks by removing all non-duckie pixels. Also, each blob must be a different value
# use the function from question 1 that maps masks with single-digit pixel values to visual masks to debug

# Question 4: object detection NN

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# Or maybe just train on finding the centroid of the blobs, and assume all objects will be 50x50 or something like that