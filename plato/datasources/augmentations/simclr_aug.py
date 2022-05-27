import torchvision.transforms as T


class SimCLRTransform():

    def __init__(self, image_size):
        image_size = 224 if image_size is None else image_size

        s = 1
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.train_transform = T.Compose([
            T.RandomResizedCrop(size=image_size),
            T.RandomHorizontalFlip(),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
        ])

        self.test_transform = T.Compose([
            T.Resize(size=image_size),
            T.ToTensor(),
        ])

    def __call__(self, x):

        x1 = self.train_transform(x)
        x2 = self.train_transform(x)

        return x1, x2