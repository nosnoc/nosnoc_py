from typing import Optional,Union

AbstractDims = Union['Dims']

class Dims:
    r"""
     for "genericing" dims classes with the necessary attrs etc.
    """
    def __init__(self, parent: Optional[AbstractDims]=None):
        object.__setattr__(self,"parent", parent)
        return self

    def __setattr__(self, name, value):
        if self.parent is not None and name in vars(self.parent):
            object.__setattr__(object.__getattribute__(self,"parent"), name, value)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        getattr(object.__getattribute__(self,"parent"), name)

    def __repr__(self):
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))
