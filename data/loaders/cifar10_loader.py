import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl


class Cifar10Loader(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        akeys = args.keys()
        self.target_size = args['target_size'] if 'target_size' in akeys else (64, 64)
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 2
        self.classes = args['classes'] if 'classes' in akeys else ['dog']

        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        # Transforms
        TT = transforms.ToTensor()
        TPIL = transforms.ToPILImage()
        RES = transforms.Resize(self.target_size)
        GR = transforms.Grayscale(num_output_channels=1)
        self.transform_no_aug = transforms.Compose([TPIL, GR, RES, TT])
        self.classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                          'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def train_dataloader(self):
        trainset = CIFAR10(root='./data', train=True, download=True)

        datasets_array = [get_class_i(trainset.data, trainset.targets, self.classDict[d_name]) for d_name in self.classes]
        select_set = DatasetMaker(datasets_array, self.transform_no_aug)

        dataloader = DataLoader(
            select_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        testset = CIFAR10(root='./data', train=False, download=True)

        datasets_array = [get_class_i(testset.data, testset.targets, self.classDict[d_name]) for d_name in self.classes]
        select_set = DatasetMaker(datasets_array, self.transform_no_aug)

        dataloader = DataLoader(
            select_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class



