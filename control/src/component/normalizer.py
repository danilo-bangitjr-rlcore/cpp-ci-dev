
class BaseNormalizer:
    def __init__(self):
        return

    def __call__(self, x):
        return x


class Identity(BaseNormalizer):
    def __init__(self):
        super(Identity, self).__init__()
        

def init_normalizer(name, info):
    if name == "Identity":
        return Identity()
    else:
        raise NotImplementedError