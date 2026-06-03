import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(
    batch_size: int = 128,
    seed: int = 0,
    data_root: str = "./data",
    augment_train: bool = True,
    num_workers: int = 2,
):
    """
    Returns (train_loader, val_loader, test_loader).

    Split: 45 000 train / 5 000 val / 10 000 test.
    Augmentation is applied to train only when augment_train=True (ResNet).
    MLP should call with augment_train=False since it flattens the input anyway.
    """
    normalize = transforms.Normalize(_MEAN, _STD)

    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    full_train = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=test_transform)

    # Deterministic 45k / 5k split
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [45000, 5000], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
