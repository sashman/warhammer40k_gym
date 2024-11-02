import numpy as np

def get_direction_from_polar(angle_radians, magnitude):
    x = magnitude * np.cos(angle_radians)
    y = magnitude * np.sin(angle_radians)
    direction = np.floor(np.array([x,y])).astype(int)

    return direction