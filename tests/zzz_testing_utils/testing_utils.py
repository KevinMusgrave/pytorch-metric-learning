import numpy as np

def is_not_none_if_condition(cls, x, y, condition):
    for i in y:
        attr = getattr(x, i, None)
        if condition:
            cls.assertTrue(attr is not None)
        else:
            cls.assertTrue(attr is None)


def angle_to_coord(angle, normalized=False):
    # multiply to unnormalize
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    if not normalized:
        x *= 2
        y *= 0.5
    return x, y