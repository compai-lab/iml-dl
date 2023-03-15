from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import AddChannelIfNeeded, AssertChannelFirst, ReadImage, To01


class MVTecDataset(DefaultDataset):

    def __init__(self, data_dir, file_type='', label_dir=None, target_size=(64, 64)):
        """
        @param data_dir: str
            path to directory or csv file containing data
        @param: file_type: str
            ending of the files, e.g., '*.jpg'
        @param: label_dir: str
            path to directory or csv file containing labels
        @param: image_transform: transform function, default: None
            image transforms, e.g., loading, resize, etc...
        @param: label_transform: transform function, default: None
            label_transform, e.g., loading, resize, etc...
        @param: target_size: tuple (int, int), default: (64, 64)
            the desired output size
        """
        super(MVTecDataset, self).__init__(data_dir, file_type, label_dir, target_size)

    def get_image_transform(self):
        TPIL = transforms.ToPILImage()
        TT = transforms.ToTensor()
        RES = transforms.Resize(self.target_size)
        Gray = transforms.Grayscale(num_output_channels=1)
        default_t = transforms.Compose([ReadImage(), To01(), TPIL, RES, TT])
        return default_t

    def get_label_transform(self):
        TPIL = transforms.ToPILImage()
        TT = transforms.ToTensor()
        RES = transforms.Resize(self.target_size)
        Gray = transforms.Grayscale()
        default_t = transforms.Compose([ReadImage(), To01(), TPIL, RES, TT])
        return default_t