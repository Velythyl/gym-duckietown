import cv2
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform(observations, expert_actions=None, input_shape=(120,160), as_tensor=False, on_device=True, only_obs=False): # TODO propagate input shape etc
    # Resize images
    observations = [Image.fromarray(cv2.resize(observation, dsize=input_shape[::-1])) for observation in
                    observations]
    # Transform to tensors
    compose_obs = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # using imagenet normalization values
    ])

    observations = [compose_obs(observation).numpy() for observation in observations]

    if as_tensor:
        observations = torch.tensor(observations)

    if as_tensor and on_device:
        observations = observations.to(device)

    if only_obs:
        return observations

    try:
        # scaling velocity to become in 0-1 range which is multiplied by max speed to get actual vel
        # also scaling steering angle to become in range -1 to 1 to make it easier to regress
        expert_actions = [np.array([expert_action[0], expert_action[1] / (np.pi / 2)]) for expert_action in
                          expert_actions]
    except:
        pass
    expert_actions = [torch.tensor(expert_action).numpy() for expert_action in expert_actions]

    if as_tensor:
        expert_actions = torch.tensor(expert_actions).to(device)

    return observations, expert_actions
