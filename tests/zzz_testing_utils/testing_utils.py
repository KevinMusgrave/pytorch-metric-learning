import numpy as np


def angle_to_coord(angle, normalized=False):
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    if not normalized:
        x *= 2
        y *= 0.5
    return x, y
