import math
import random

import numpy as np
from PIL import Image
import cv2

from gym_duckietown.simulator import _actual_center

NORMALIZE_TO_NB_OF_RED_PIXELS = 2000
EPSILON_AVG_RED = 50
SIZE_SIDE_FINAL_IMG = 125

def count_red_pixels(hsv_img):
    return cv2.countNonZero(
        cv2.cvtColor(
            cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            , cv2.COLOR_BGR2GRAY
        )
    )

def direction(red_only_img):
    current_nb_of_red_pixels = count_red_pixels(red_only_img)
    # -1 if downsize, +1 if upsize, 0 if done
    if current_nb_of_red_pixels <= NORMALIZE_TO_NB_OF_RED_PIXELS - EPSILON_AVG_RED:
        return 1
    if current_nb_of_red_pixels >= NORMALIZE_TO_NB_OF_RED_PIXELS + EPSILON_AVG_RED:
        return -1
    return 0

class Cache:
    def __init__(self):
        self.cache = {}
        self.capacity = 2000
    def add(self, nb_red_pixels, shape):
        if nb_red_pixels in self.cache:
            return
        else:
            if len(self.cache) >= self.capacity:
                key = random.choice(list(self.cache.keys()))
                del self.cache[key]
            self.cache[nb_red_pixels] = shape
    def __getitem__(self, nb_red_pixels):
        return self.cache[nb_red_pixels]
    def __contains__(self, item):
        return item in self.cache

red_pixel_cache = Cache()

# TODO we'll have to make a NN that does this job. Use this function to generate the resizing it should do + the position of the car
def isolate_car(image, env):
    image = image.astype("uint8")
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img = cv2.copyMakeBorder(img, SIZE_SIDE_FINAL_IMG, SIZE_SIDE_FINAL_IMG, SIZE_SIDE_FINAL_IMG, SIZE_SIDE_FINAL_IMG, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Match the car
    # https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
    lower_red = np.array([0, 200, 200])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(img, lower_red, upper_red)
    red_only_img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # We want to normalize the images so that the car is always approx the same number of pixels
    # Here, we play with the red only image until we find a width a height that are appropriate.
    # This makes the image blurry because of the chained resizing
    initial_red_pixel_count = count_red_pixels(red_only_img)

    if initial_red_pixel_count in red_pixel_cache:
        tup = red_pixel_cache[initial_red_pixel_count]
        red_only_img = cv2.resize(red_only_img, tup)
        print("used_cache")

    if direction(red_only_img) != 0:
        scale = NORMALIZE_TO_NB_OF_RED_PIXELS / count_red_pixels(red_only_img)
        width = int(red_only_img.shape[1] * math.sqrt(scale))
        height = int(red_only_img.shape[0] * math.sqrt(scale))
        red_only_img = cv2.resize(red_only_img, (width, height))
        initial_direction = direction(red_only_img)
        while direction(red_only_img) == initial_direction and initial_direction != 0:
            width += initial_direction * 20
            height += initial_direction * 20
            red_only_img = cv2.resize(red_only_img, (width, height))

        # New that we know the right width and height, we can resize the actual image one single time!
        red_pixel_cache.add(initial_red_pixel_count, (width, height))
    img = cv2.resize(img, (red_only_img.shape[1], red_only_img.shape[0]))

    # Finally, we want to isolate the actual car
    # To do this, let's find the center of the blob that's the car in the red_only_image
    # https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    binary_red_img = cv2.cvtColor(
        cv2.cvtColor(red_only_img, cv2.COLOR_HSV2BGR)
        , cv2.COLOR_BGR2GRAY
    )

    ret, thresh = cv2.threshold(binary_red_img, 1, 255, 0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    x = int(cX - SIZE_SIDE_FINAL_IMG/2)
    y = int(cY - SIZE_SIDE_FINAL_IMG/2)

    img = img[y:y+SIZE_SIDE_FINAL_IMG, x:x+SIZE_SIDE_FINAL_IMG]

    return img
