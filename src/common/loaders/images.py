import math
import gzip
import random
import codecs
import torch.utils.data as data
from PIL import Image
import errno
import os
import os.path
import numpy as np
from torchvision.datasets.utils import download_url, makedir_exist_ok
from torch.utils.model_zoo import tqdm
import torch
from torchvision import datasets, transforms
from skimage import transform, filters
from quickdraw import QuickDrawData


def cifar10(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.classes))

    return train_loader, test_loader, shape, n_classes


class Rotate(object):
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        d = random.randrange(-30, 30)
        img = transform.rotate(img, d, mode='edge', order=4)
        return img


class Jitter(object):
    def __call__(self, img):
        if random.random() > 0.75:
            return img
        img = transforms.ColorJitter((0.5, 1), (0.5, 1), 1, 0.5)(img)
        return img


class BW(object):
    def __call__(self, img):
        if random.random() > 0.5:
            return img
        gray = img.convert('L')
        #gray = filters.sobel(gray)
        #gray = Image.fromarray(np.uint8(gray * 255), 'L')
        #bw = gray.point(lambda x: 0 if x < 128 else 255, '1').convert('RGB')
        return gray


class Rescale(object):
    def __call__(self, img):
        img = transforms.RandomResizedCrop(img.size[1], scale=(0.5, 1.5))(img)
        return img


class Occlude(object):
    def __call__(self, img):
        #img = np.asarray(img, np.uint8)
        size = math.ceil(img.shape[0] / 4)
        mask = np.ones_like(img)
        if random.random() > 0.5:
            x = int(img.shape[0] / 4 * 3)
            mask[:,x:x + size] *= 0
        if random.random() > 0.5:
            mask[:,:size] *= 0
        img = img * mask
        return img


class Sobel(object):
    def __call__(self, img):
        if random.random() > 0.5:
            img = filters.sobel(img)
            #img = img.astype(np.float)
        else:
            img = np.asarray(img, np.float)
        img /= img.max()
        return img



def svhn(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose((
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.labels))

    return train_loader, test_loader, shape, n_classes


def bsvhn(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose((
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
    ))
    train = BSVHN(root, split='train', download=True, transform=transform)
    test = BSVHN(root, split='test', download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.labels))

    return train_loader, test_loader, shape, n_classes


def triple_channel(x):
    if x.shape[0] == 3:
        return x
    return torch.cat((x,x,x), 0)


def mnist(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.train_labels.tolist()))

    return train_loader, test_loader, shape, n_classes


def visda(root, train_batch_size, test_batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=1),
        transforms.ToTensor(),
        triple_channel,
    ])
    train = datasets.ImageFolder(root, transform=transform)
    test = datasets.ImageFolder(root, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.classes))

    return train_loader, test_loader, shape, n_classes


def ivisda(root, train_batch_size, test_batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=1),
        transforms.ToTensor(),
        triple_channel,
    ])
    train = datasets.ImageFolder(root, transform=transform)
    n_classes = len(set(train.classes))
    t = [
        transforms.RandomAffine(30, (0, 0), (0.8, 1.2), 30, fillcolor=(255,255,255)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(1),
        Jitter(),
        #Sobel(),
        transforms.ToTensor(),
        triple_channel,
    ]

    train = MultiTransformDataset(train, t)
    test = datasets.ImageFolder(root, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=24)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=24)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, shape, n_classes


def quickdraw(root, train_batch_size, test_batch_size, **kwargs):
    classes = ['t-shirt', 'pants', 'shoe', 'purse']
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=1),
        transforms.ToTensor(),
    ])
    train = QuickDrawDataset(root, classes, transform)
    test = QuickDrawDataset(root, classes, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.classes))

    return train_loader, test_loader, shape, n_classes


def iquickdraw(root, train_batch_size, test_batch_size, **kwargs):
    classes = ['t-shirt', 'pants', 'shoe', 'purse']
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=1),
        transforms.ToTensor(),
    ])
    train = QuickDrawDataset(root, classes, transform)
    n_classes = len(set(train.classes))
    t = [
        transforms.RandomAffine(30, (0, 0), (0.8, 1.2), 30, fillcolor=(255,255,255)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]
    train = MultiTransformDataset(train, t)
    test = QuickDrawDataset(root, classes, transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, shape, n_classes


def fashion(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        triple_channel,
    ])
    train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=transform)

    classes = [0, 1, 7, 8]
    train = FilterDataset(train, classes)
    test = FilterDataset(test, classes)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.classes))
    return train_loader, test_loader, shape, n_classes


def inverse(x):
    return 1-x


def omniglot(root, train_batch_size, test_batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        inverse,
        triple_channel,
    ])
    train = datasets.ImageFolder(root, transform=transform)
    n_classes = 100
    t = [transforms.RandomAffine(30, (0, 0), (0.7, 1.3), 40), transforms.ToTensor()]
    train = MultiTransformDataset(train, t)
    test = datasets.ImageFolder(root, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, shape, n_classes


def imnist(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
    ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    n_classes = len(set(train.train_labels.tolist()))
    t = [transforms.RandomAffine(30, (0, 0), (0.5, 1.5), 40), transforms.ToTensor()]
    train = MultiTransformDataset(train, t)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader, shape, n_classes


def isvhn(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose((
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
    ))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    n_classes = len(set(train.labels))
    shape = train[0][0].shape
    t = [Rescale(), Jitter(), Rotate(), transforms.ToTensor(), triple_channel]
    train = MultiTransformDataset(train, t)
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)


    return train_loader, test_loader, shape, n_classes


def icifar10(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(root, train=True, download=True, transform=transform)
    n_classes = len(set(train.classes))
    shape = train[0][0].shape
    t = [Rescale(), Jitter(), Rotate(), transforms.ToTensor(), triple_channel]
    train = MultiTransformDataset(train, t)
    test = datasets.CIFAR10(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    return train_loader, test_loader, shape, n_classes


def icifar100(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.ToTensor()
    train = datasets.CIFAR100(root, train=True, download=True, transform=transform)
    n_classes = len(set(train.classes))
    shape = train[0][0].shape
    t = [Rescale(), Jitter(), Rotate(), transforms.ToTensor(), triple_channel]
    train = MultiTransformDataset(train, t)
    test = datasets.CIFAR100(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    return train_loader, test_loader, shape, n_classes


def rmnist(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.RandomRotation((90, 90)),
        transforms.ToTensor(),
        triple_channel,
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.MNIST(root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.train_labels.tolist()))

    return train_loader, test_loader, shape, n_classes


def bmnist(root, train_batch_size, test_batch_size=None, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
        triple_channel,
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = BMNIST(root, train=True, download=True, transform=transform)
    test = BMNIST(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.train_labels.tolist()))

    return train_loader, test_loader, shape, n_classes


def usps(root, train_batch_size, test_batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
    ])
    train = USPS(root, split='train', download=True, transform=transform)
    test = USPS(root, split='test', download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.labels))

    return train_loader, test_loader, shape, n_classes


def mnistm(root, train_batch_size, test_batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
    ])
    train = MNISTM(root, train=True, download=True, transform=transform)
    test = MNISTM(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.train_labels.tolist()))

    return train_loader, test_loader, shape, n_classes


def mnistc(root, train_batch_size, test_batch_size, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(32, interpolation=0),
        transforms.ToTensor(),
    ])
    train = MNISTC(root, train=True, download=True, transform=transform)
    test = MNISTC(root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size or train_batch_size,
                                              shuffle=True, num_workers=8)

    shape = train_loader.dataset[0][0].shape
    n_classes = len(set(train.train_labels.tolist()))

    return train_loader, test_loader, shape, n_classes


class FilterDataset(data.Dataset):
    def __init__(self, dataset, classes):
        filtered_dataset = list(filter(lambda x: x[1] in classes, dataset))
        images = list(map(lambda x: x[0], filtered_dataset))
        labels = [classes.index(data[1]) for data in filtered_dataset]

        self.tensor = images
        self.classes = labels

    def __getitem__(self, idx):
        input = self.tensor[idx]
        target = self.classes[idx]
        return input, target

    def __len__(self):
        return len(self.tensor)


class MultiTransformDataset(data.Dataset):
    def __init__(self, dataset, t):
        self.dataset = dataset
        self.transform = transforms.Compose(
            [transforms.ToPILImage()] +
            t)

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        return input, self.transform(input), target

    def __len__(self):
        return len(self.dataset)


class QuickDrawDataset(data.Dataset):
    def __init__(self, root, classes, transform):
        self.classes = classes
        self.labels = torch.arange(len(classes))
        self.transform = transform
        self.qdd = QuickDrawData(recognized=True, max_drawings=10000, cache_dir=root)
        self.qdd.load_drawings(classes)

    def __getitem__(self, idx):
        c = self.classes[idx%len(self.classes)]
        label = self.labels[idx%len(self.classes)]
        img = self.qdd.get_drawing(c).image
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return 10000
        #return len(self.classes)


class USPS(data.Dataset):
    """`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.
    The data-format is : [label [index:value ]*256 \n] * num_lines, where ``label`` lies in ``[1, 10]``.
    The value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``
    and make pixel values in ``[0, 255]``.
    Args:
        root (string): Root directory of dataset to store``USPS`` data files.
        split (string): One of {'train', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    split_list = {
        'train': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
            "usps.bz2", 7291
        ],
        'test': [
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
            "usps.t.bz2", 2007
        ],
    }

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.total_images = self.split_list[split][2]

        full_path = os.path.join(self.root, self.filename)

        if download and not os.path.exists(full_path):
            self.download()

        import bz2
        fp = bz2.open(full_path)

        datas = []
        targets = []
        for line in tqdm(
                fp, desc='processing data', total=self.total_images):
            label, *pixels = line.decode().split()
            pixels = [float(x.split(':')[-1]) for x in pixels]
            im = np.asarray(pixels).reshape((16, 16))
            im = (im + 1) / 2 * 255
            im = im.astype(dtype=np.uint8)
            datas.append(im)
            targets.append(int(label) - 1)

        assert len(targets) == self.total_images, \
            'total number of images are wrong! maybe the download is corrupted?'

        self.data = np.stack(datas, axis=0)
        self.targets = targets
        self.labels = list(range(10))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        download_url(self.url, self.root, self.filename, md5=None)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return fmt_str


class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, mnist_root="data", train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file)
            )
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file)
            )

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print("Downloading " + self.url)
        filename = self.url.rpartition("/")[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace(".gz", "")):
            data = urllib.request.urlopen(self.url)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print("Processing...")

        # load MNIST-M images from pkl file
        with open(file_path.replace(".gz", ""), "rb") as f:
            mnist_m_data = pickle.load(f, encoding="bytes")
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b"train"])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b"test"])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root, train=True, download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root, train=False, download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root, self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")


class BSVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`
    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.labels_idx = [np.where(self.labels == i)[0] for i in range(10)]
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        np.random.seed()
        label = np.random.randint(0, 10)
        idxes = self.labels_idx[label]
        index = np.random.choice(idxes)

        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class MNISTC(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    targets_color = {0: ((0,1), (0,0), (0,0)),
                     1: ((0,0), (0,1), (0,0)),
                     2: ((0,0), (0,0), (0.5,1)),
                     3: ((0.5,1), (0.5,1), (0,0)),
                     4: ((0.5,1), (0,0), (0.5,1)),
                     5: ((0,0), (0.5,1), (0.5,1)),
                     6: ((0,1), (1,1), (1,1)),
                     7: ((1,1), (0,1), (1,1)),
                     8: ((1,1), (1,1), (0,1)),
                     9: ((1,1), (1,1), (1,1))}

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.labels_idx = [np.where(self.targets == i)[0] for i in range(10)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #label = np.random.randint(0, 2)
        #idxes = self.labels_idx[label]
        #index = np.random.choice(idxes)
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = torch.cat((img, img, img))
        color = self.targets_color[target]
        #r = random.uniform(color[0][0], color[0][1])
        #g = random.uniform(color[1][0], color[1][1])
        #b = random.uniform(color[2][0], color[2][1])
        r = random.uniform(0.1, 1)
        g = random.uniform(0.1, 1)
        b = random.uniform(0.1, 1)
        #img[1] = 0
        #img[2] = 0
        img[0] = img[0] * r
        img[1] = img[1] * g
        img[2] = img[2] * b

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class BMNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.labels_idx = [np.where(self.targets == i)[0] for i in range(10)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        np.random.seed()
        label = np.random.randint(0, 10)
        idxes = self.labels_idx[label]
        index = np.random.choice(idxes)
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
            gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
