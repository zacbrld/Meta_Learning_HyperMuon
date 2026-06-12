__all__ = ["get_cifar10_loaders", "CSVLogger"]


def __getattr__(name):
    if name == "get_cifar10_loaders":
        from .data import get_cifar10_loaders

        return get_cifar10_loaders
    if name == "CSVLogger":
        from .logger import CSVLogger

        return CSVLogger
    raise AttributeError(name)
