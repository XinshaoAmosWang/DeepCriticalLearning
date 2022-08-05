import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# mean and std
TRAIN_MEAN = (0.6959, 0.6537, 0.6371)
TRAIN_STD = (0.3113, 0.3192, 0.3214)


def clothing1m_transform_train_rrcsr(params=None):
    if params is None:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.7777777, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),  # flip
                transforms.RandomRotation(15),  # rotation
                transforms.ToTensor(),
                transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(params["min_scale"], 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),  # flip
                transforms.RandomRotation(params["rotation"]),  # rotation
                transforms.ToTensor(),
                transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
            ]
        )


clothing1m_transform_train_rc = transforms.Compose(
    [
        transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),  # flip
        transforms.RandomRotation(15),  # rotation
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
    ]
)

clothing1m_transform_test_resizeonly = transforms.Compose(
    [
        transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
    ]
)
clothing1m_transform_test_resizecrop = transforms.Compose(
    [
        transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
    ]
)
