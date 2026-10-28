import math


def angle_to_coord(angle, normalized=False):
    radians = math.radians(angle)
    x = math.cos(radians)
    y = math.sin(radians)
    if not normalized:
        x *= 2
        y *= 0.5
    return x, y
