def is_not_none_if_condition(cls, x, y, condition):
    for i in y:
        attr = getattr(x, i, None)
        if condition:
            cls.assertTrue(attr is not None)
        else:
            cls.assertTrue(attr is None)
