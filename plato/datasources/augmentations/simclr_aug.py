"""
The implementation of the SimCLR's [1] augmentation function.

The official code: https://github.com/google-research/simclr

The third-party code: https://github.com/PatrickHua/SimSiam


Reference:

[1]. https://arxiv.org/abs/2002.05709
"""

import torchvision.transforms as T


class SimCLRTransform():
    """ This the contrastive data augmentation used by the SimCLR method. """

    def __init__(self, image_size, normalize):
        image_size = 224 if image_size is None else image_size

        s = 1
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transform_functions = [
            T.RandomResizedCrop(size=image_size),
            T.RandomHorizontalFlip(),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
        ]

        if normalize is not None:
            transform_functions.append(T.Normalize(*normalize))

        self.transform = T.Compose(transform_functions)

    def __call__(self, x):
        """ Perform the contrastive data augmentation. """
        x1 = self.transform(x)
        x2 = self.transform(x)

        return x1, x2