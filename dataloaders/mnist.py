from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import os.path as osp
import numpy as np


class MNIST(Dataset):
    """
    A customized data loader for MNIST.
    """

    def __init__(self,
                 root,
                 transform=None,
                 preload=False,
                 is_sample=False):
        """ Intialize the MNIST dataset

        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.is_sample = is_sample

        # read filenames
        for i in range(10):
            filenames = glob.glob(osp.join(root, str(i), '*.png'))
            for fn in filenames:
                if (self.is_sample is False) or (int(fn) % 10 == 0):
                    self.filenames.append((fn, i))  # (filename, label) pair

        # if preload dataset into memory
        if preload:
            self._preload()

        self.len = len(self.filenames)

    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:
            # load images
            image = Image.open(image_fn)
            self.images.append(image.copy())
            # avoid too many opened files bug
            image.close()
            self.labels.append(label)

    def as_numpy(self):
        img = []
        for image in self.images:
            img.append(np.asarray(image))
        return np.array(img), np.array(self.labels)

    # probably the most important to customize.
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)

        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return torch.rand(1, 30,30), randint(0, 9)
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def create_dataset(path, batch_size):
    trainset = MNIST(
        root=path,
        # root='/Users/shiran.s/dev/p300_net/data/mnist/training',
        preload=True, transform=transforms.ToTensor(),
    )

    print(trainset.len)
    # Use the torch dataloader to iterate through the dataset
    # We want the dataset to be shuffled during training.
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)


def build_dataloader(root_path, batch_size=64):
    print("using root path: ", root_path)

    root = root_path + '/training'
    trainset = create_dataset(root, batch_size)
    root = root_path + '/testing'
    testset = create_dataset(root, batch_size)
    return trainset, testset


def build_numpy(root_path, is_sample):
    print("using root path: ", root_path)

    root = root_path + '/training'
    trainset = MNIST(
        root, preload=True, transform=transforms.ToTensor(), is_sample=is_sample
    )
    root = root_path + '/testing'
    testset = MNIST(
        root, preload=True, transform=transforms.ToTensor(), is_sample=is_sample
    )
    return trainset.as_numpy(), testset.as_numpy()
