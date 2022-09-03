from . import common_functions as c_f


class KeyCheckerDict:
    def __init__(self, children):
        self.children = children

    def __getitem__(self, key):
        return self.children[key]

    def __setitem__(self, key, value):
        self.children[key] = value

    def verify(self, obj):
        for k, v in self.children.items():
            self._verify_prop(getattr(obj, k, None), k, v)

    def _verify_prop(self, obj, obj_name, s):
        val = lambda x: x(s, self.children) if callable(x) else x  # noqa: E731

        if s.warn_empty and obj in [None, {}]:
            c_f.LOGGER.warning("%s is empty" % obj_name)
        if obj is not None:
            keys = val(s.keys)
            for k in obj.keys():
                assert any(
                    pattern.match(k) for pattern in c_f.regex_wrapper(keys)
                ), "%s keys must be one of %s" % (obj_name, ", ".join(keys))
            for imp_key in val(s.important):
                if not any(c_f.regex_wrapper(imp_key).match(k) for k in obj):
                    c_f.LOGGER.warning('%s is missing "%s"' % (obj_name, imp_key))
            for ess_key in val(s.essential):
                assert any(
                    c_f.regex_wrapper(ess_key).match(k) for k in obj
                ), '%s must contain "%s"' % (obj_name, ess_key)


def default_important(s, d):
    return c_f.exclude(s.keys, s.essential)


# We can make this a dataclass, if we target 3.7+
class KeyChecker:
    def __init__(
        self,
        keys,
        warn_empty=True,
        important=default_important,
        essential=None,
    ):
        self.keys = keys
        self.warn_empty = warn_empty
        self.important = important
        self.essential = essential
        if self.essential is None:
            self.essential = []
