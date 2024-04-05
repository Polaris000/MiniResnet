import torch
import torchvision.transforms as transforms
from torchvision import datasets
import pickle
from PIL import Image


class TestData(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        self.data = None
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
            self.images = self.data[b"data"]
            self.ids = self.data[b"ids"]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img = self.images[index]
        id_ = self.ids[index]

        # Convert image to PIL Image
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return id_, img


def load_data(input_dim=(3, 32, 32)):
    import os

    data_directory = os.path.dirname(__file__) + "/../data"
    test_path = os.path.join(data_directory, "testdata", "cifar_test_nolabels.pkl")

    transform_train, transform_val_test = augment_data(input_dim)

    trainset = datasets.CIFAR10(
        root=data_directory, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    val_set = datasets.CIFAR10(
        root=data_directory, train=False, download=True, transform=transform_val_test
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=100, shuffle=False, num_workers=2
    )

    test_set = TestData(file_path=test_path, transform=transform_val_test)
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, val_loader, testloader, classes


def augment_data(input_dim=(3, 32, 32)):
    transform_train = transforms.Compose(
        [
            # transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.RandomCrop(input_dim[1], padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Grayscale(num_output_channels=1),
        ]
    )

    transform_val_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Grayscale(num_output_channels=1),
        ]
    )
    return transform_train, transform_val_test
