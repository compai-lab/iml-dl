from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import AddChannelIfNeeded, AssertChannelFirst, ReadImage, To01
# from transforms.preprocessing import Pad, Slice
# from transforms.preprocessing import Binarize, AdjustIntensity


class TacDataset(DefaultDataset):
    # def __init__(self, data_dir, file_type='', label_dir=None, target_size=(64, 64), test=False):
    def __init__(self, data_dir, file_type='', label_dir=None, mask_dir=None, target_size=(256, 256), test=False):

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
        super(TacDataset, self).__init__(data_dir, file_type, label_dir, mask_dir, target_size, test)

    def get_image_transform(self):
        TPIL = transforms.ToPILImage()
        TT = transforms.ToTensor()
        RES = transforms.Resize(self.target_size)
        Gray = transforms.Grayscale(num_output_channels=1) #Pad((0,0,125, 125)),
        CenterCrop = transforms.CenterCrop(512)
        CJ = transforms.ColorJitter(brightness=0.3, contrast=0.3)
        RandRot = transforms.RandomAffine(degrees=(-10, 10), translate=(0.01, 0.01))

        default_t = transforms.Compose([ReadImage(), TPIL, Gray, RES, TT])
        return default_t

    def get_label_transform(self):
        TPIL = transforms.ToPILImage()
        TT = transforms.ToTensor()
        RES = transforms.Resize(self.target_size)
        Gray = transforms.Grayscale()
        CenterCrop = transforms.CenterCrop(512)
        # TBIN = Binarize(0)
        default_t = transforms.Compose([ReadImage(), To01(), TPIL, RES, TT])
        return default_t