# for testing stuff, prototyping, etc
import cv2
import numpy as np

obs_arr = np.load("./dataset/obs.npy")
rew_arr = np.load("./dataset/rew.npy")
cv2.imshow("", obs_arr[0])
cv2.waitKey()