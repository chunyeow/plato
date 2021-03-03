"""
The CIFAR-10 dataset.
"""

from torchvision import datasets, transforms

from datasources import base


class DataSource(base.DataSource):
    """The CIFAR-10 dataset."""
    def __init__(self, path):
        super().__init__(path)

        _transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.trainset = datasets.CIFAR10(root=self._path,
                                         train=True,
                                         download=True,
                                         transform=_transform)
        self.testset = datasets.CIFAR10(root=self._path,
                                        train=False,
                                        download=True,
                                        transform=_transform)

    def num_train_examples(self):
        return 50000

    def num_test_examples(self):
        return 10000
