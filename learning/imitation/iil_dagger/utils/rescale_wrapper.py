from gym import ActionWrapper
import numpy as np


class RescaleActionWrapper(ActionWrapper):
    def __init__(self, env, max_velocity, steering_gain):
        super(ActionWrapper, self).__init__(env)
        self.max_velocity = max_velocity
        self.steering_gain = steering_gain

    def action(self, action):
        action = np.clip(action, -1, 1) # Sometimes, patch+combined is less than one (can actually be -2 to +2, technically)

        action_ = [self.max_velocity*(action[0]--1)/(1--1), action[1]*self.steering_gain]

        print(action_)
        return action_