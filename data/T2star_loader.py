import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import h5py
import logging
from typing import NamedTuple
import nibabel as nib
import glob
from medutils.mri import ifft2c, rss


class RawT2starDataset(Dataset):
    """Dataset for loading raw T2* data

    Parameters
    ----------
    path : str
        path to folder containing the relevant h5 files
    only_bm_slices : bool
        whether slices with percentage of brainmask voxels < bm_thr*100% should
        be excluded or not
    bm_thr : float
        threshold for including / excluding slices based on percentage of
        brainmask voxels
    normalize : str
        whether to normalize the data with maximum absolute value of image
        ('abs_image')
    select_echo : bool or int
        whether to select a specific echo.
    random mask : bool or list
        whether to generate a random mask. If yes, then provide a list [R, N]
        with acceleration rate R and number of central lines N.
    overfit_one_sample : bool
        whether only one sample should be loaded to test overfitting
    """

    def __init__(self, path, only_bm_slices=False, bm_thr=0.1,
                 normalize="abs_image", select_echo=False,
                 random_mask=[2, 10], overfit_one_sample=False):
        super().__init__()

        self.path = path
        self.raw_samples = []
        self.only_bm_slices = only_bm_slices
        self.bm_thr = bm_thr
        self.normalize = normalize
        self.overfit_one_sample = overfit_one_sample
        self.select_echo = select_echo
        self.random_mask = random_mask

        files = list(pathlib.Path(self.path).iterdir())
        for filename in sorted(files):
            slices_ind = self._get_slice_indices(filename)

            new_samples = []
            for slice_ind in slices_ind:
                new_samples.append(T2StarDataSample(filename, slice_ind))

            self.raw_samples += new_samples

    def _get_slice_indices(self, filename):
        with h5py.File(filename, "r") as hf:
            if self.only_bm_slices and not self.overfit_one_sample:
                brain_mask = hf["Brain_mask"]  # Brain mask shape: nr_slices, PE_lines, readout
                bm_summed = np.sum(brain_mask, axis=(1, 2))
                slices_ind = np.where(bm_summed / (brain_mask.shape[1] * brain_mask.shape[2]) > self.bm_thr)[0]
            elif self.overfit_one_sample:
                slices_ind = [self.overfit_one_sample]
            else:
                slices_ind = np.arange(hf['out']['Data'].shape[0])
        return slices_ind

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        filename, dataslice = self.raw_samples[idx]

        with h5py.File(filename, "r") as f:
            raw_data = f['out']['Data'][dataslice, :, 0, 0, :, 0]
            tmp = f['out']['Parameter']['YRange'][:]
            if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
                print('Error: different y shifts for different echoes!')
            y_shift = -int((tmp[0, 0] + tmp[1, 0]) / 2)
            # y_shift to be used on images (see below)

            # convert to proper complex data
            if isinstance(raw_data, np.ndarray) and raw_data.dtype == [('real', '<f4'), ('imag', '<f4')]:
                kspace = raw_data.view(np.complex64).astype(np.complex64)
                sens_maps = f['out']['SENSE']['maps'][dataslice, :, 0, 0, :, 0].view(np.complex64).astype(np.complex64)
            else:
                print('Error in load_raw_mat: Unexpected data format: ',
                      raw_data.dtype)

        # undersample with a random mask:
        nc, ne, npe, nfe = kspace.shape
        if self.random_mask is not False:
            mask = np.random.choice([1, 0], (npe), p=[1 / self.random_mask[0], 1 - 1 / self.random_mask[0]])
            mask[npe//2 - self.random_mask[1]//2:npe//2 + self.random_mask[1]//2] = 1
        else:
            # mask = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0,
            #                  1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,
            #                  1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
            #                  1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            #                  1, 1, 0, 1])
            mask = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1,
                             1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                             1, 1, 1, 0])
        mask = mask.reshape(1, 1, npe, 1).repeat(nc, axis=0).repeat(ne, axis=1).repeat(nfe, axis=3)
        kspace_zf = kspace*mask

        # pad coil sensitivity maps to have same shape as images:
        pad_width = ((0, 0), (0, 0), (0, 0), (int((kspace.shape[-1] - sens_maps.shape[-1])/2), int((kspace.shape[-1] - sens_maps.shape[-1])/2)))
        sens_maps = np.pad(sens_maps, pad_width, mode='constant')
        sens_maps = np.nan_to_num(sens_maps / rss(sens_maps, 1))

        # zero-filled and fully sampled coil combined reconstructions:
        coil_imgs_fs = ifft2c(kspace)
        coil_imgs_fs = np.roll(coil_imgs_fs, shift=y_shift, axis=-2)
        img_cc_fs = np.sum(coil_imgs_fs*np.conj(sens_maps), axis=1)
        coil_imgs_zf = ifft2c(kspace_zf)
        coil_imgs_zf = np.roll(coil_imgs_zf, shift=y_shift, axis=-2)
        img_cc_zf = np.sum(coil_imgs_zf * np.conj(sens_maps), axis=1)

        if self.select_echo is not False:
            img_cc_fs = np.array([img_cc_fs[self.select_echo]])
            img_cc_zf = np.array([img_cc_zf[self.select_echo]])
            mask = np.array([mask[self.select_echo]])

        if self.normalize == "abs_image":
            norm = np.nanmax(abs(img_cc_zf)) + 1e-9
            img_cc_zf /= norm
            img_cc_fs /= norm

        # equal coil dimension of 32 for all datasets:
        sens_maps_32 = np.zeros((sens_maps.shape[0], 32, sens_maps.shape[2], sens_maps.shape[3]), dtype=sens_maps.dtype)
        sens_maps_32[:, :sens_maps.shape[1]] = sens_maps
        mask_32 = np.zeros((mask.shape[0], 32, mask.shape[2], mask.shape[3]), dtype=mask.dtype)
        mask_32[:, :mask.shape[1]] = mask

        # return zero-filled image, fully sampled image, mask sensitivity maps etc
        return torch.as_tensor(img_cc_zf, dtype=torch.complex64), \
               torch.as_tensor(img_cc_fs, dtype=torch.complex64), \
               torch.as_tensor(mask_32, dtype=torch.complex64), \
               torch.as_tensor(sens_maps_32, dtype=torch.complex64), \
               str(filename), dataslice


class RawT2starLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        akeys = args.keys()
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.only_brainmask_slices = args['only_brainmask_slices'] if 'only_brainmask_slices' in akeys else False
        self.bm_thr = args['bm_thr'] if 'bm_thr' in akeys else 0.1
        self.normalize = args['normalize'] if 'normalize' in akeys else "abs_image"
        self.overfit_one_sample = args['overfit_one_sample'] if 'overfit_one_sample' in akeys else False
        self.select_echo = args['select_echo'] if 'select_echo' in akeys else False
        self.random_mask = args['random_mask'] if 'random_mask' in akeys else [2, 10]
        self.drop_last = False if self.overfit_one_sample else True
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 1
        self.data_dir = args['data_dir'] if 'data_dir' in akeys else None
        assert type(self.data_dir) is dict, 'DefaultDataset::init():  data_dir variable should be a dictionary'


    def train_dataloader(self):
        """Loads a batch of training data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files.
        """
        trainset = RawT2starDataset(self.data_dir['train'],
                                    only_bm_slices=self.only_brainmask_slices,
                                    bm_thr=self.bm_thr,
                                    normalize=self.normalize,
                                    select_echo=self.select_echo,
                                    random_mask=self.random_mask,
                                    overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the train dataset: {trainset.__len__()}.")

        dataloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        """Loads a batch of validation data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files.
        """
        valset = RawT2starDataset(self.data_dir['val'],
                                  only_bm_slices=self.only_brainmask_slices,
                                  bm_thr=self.bm_thr,
                                  normalize=self.normalize,
                                  select_echo=self.select_echo,
                                  random_mask=self.random_mask,
                                  overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the validation dataset: {valset.__len__()}.")

        dataloader = DataLoader(
            valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        """Loads a batch of testing data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files. For the
        test data loader 'drop_last' is enabled, which means that no data will
        be loaded if the batch size is larger than the size of the test set.
        """
        testset = RawT2starDataset(self.data_dir['test'],
                                   only_bm_slices=self.only_brainmask_slices,
                                   bm_thr=self.bm_thr,
                                   normalize=self.normalize,
                                   select_echo=self.select_echo,
                                   random_mask=self.random_mask,
                                   overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the test dataset: {testset.__len__()}.")

        # if testset.__len__() < self.batch_size:
        #     logging.info('The batch size ({}) is larger than the size of the test set({})! Since the dataloader has '
        #                  'drop_last enabled, no data will be loaded!'.format(self.batch_size, testset.__len__()))

        dataloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader


class T2starDataset(Dataset):
    """Dataset for loading simulated T2* data

    Parameters
    ----------
    path : str
        path to folder containing the relevant h5 files
    only_bm_slices : bool
        whether slices with percentage of brainmask voxels < bm_thr*100% should
        be excluded or not
    bm_thr : float
        threshold for including / excluding slices based on percentage of
        brainmask voxels
    normalize : str
        whether to normalize the data with maximum absolute value of image
        ('abs_image') or line by line ('line_wise')
    soft_mask : bool
        whether a soft mask (values between 0 and 1 corresponding to motion
        score) should be loaded
    subst_with_orig : None or True
        whether lines with motion below a threshold should be substituted with
        motion-free data
    magn_phase : bool
        True, if magnitude and phase of the complex data to be loaded on
        separate channels, False, if real and imaginary part to be loaded
    input_2d : bool
        whether input should be loaded as 2D (echoes on different channels)
    overfit_one_sample : bool
        whether only one sample should be loaded to test overfitting
    """

    def __init__(self, path, only_bm_slices=False, bm_thr=0.1, normalize="abs_image",
                 soft_mask=False, subst_with_orig=None, crop_readout=False,
                 magn_phase=False, input_2d=False, overfit_one_sample=False):
        super().__init__()

        self.path = path
        self.raw_samples = []
        self.only_bm_slices = only_bm_slices
        self.bm_thr = bm_thr
        self.normalize = normalize
        self.crop_readout = crop_readout
        self.overfit_one_sample = overfit_one_sample
        self.magn_phase = magn_phase
        self.input_2d = input_2d
        self.soft_mask = soft_mask
        self.subst_with_orig = subst_with_orig

        files = list(pathlib.Path(self.path).iterdir())
        for filename in sorted(files):
            slices_ind = self._get_slice_indices(filename)

            new_samples = []
            for slice_ind in slices_ind:
                new_samples.append(T2StarDataSample(filename, slice_ind))

            self.raw_samples += new_samples


    def _get_slice_indices(self, filename):
        with h5py.File(filename, "r") as hf:
            if self.only_bm_slices and not self.overfit_one_sample:
                brain_mask = hf["Brain_mask"]  # Brain mask shape: nr_slices, PE_lines, readout
                bm_summed = np.sum(brain_mask, axis=(1, 2))
                slices_ind = np.where(bm_summed / (brain_mask.shape[1] * brain_mask.shape[2]) > self.bm_thr)[0]
            elif self.overfit_one_sample:
                slices_ind = [self.overfit_one_sample]
            else:
                slices_ind = np.arange(hf["Simulated_Data"].shape[1])
        return slices_ind

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        filename, dataslice = self.raw_samples[idx]

        with h5py.File(filename, "r") as hf:
            simulated_image = hf["Simulated_Data"][:, dataslice]
            if self.soft_mask and not self.subst_with_orig:
                target_mask = hf["Soft_Corruption_Mask"][:, dataslice]
                target_mask = np.mean(target_mask, axis=0)

            if not self.soft_mask and not self.subst_with_orig:
                target_mask = hf["Corruption_Mask"][:, dataslice]
                target_mask = np.mean(target_mask, axis=0)
                target_mask[target_mask <= 0.5] = 0
                target_mask[target_mask > 0.5] = 1.0

            if self.subst_with_orig:
                target_mask = hf["Soft_Corruption_Mask"][:, dataslice]
                target_mask = np.mean(target_mask, axis=0)
                original_kspace = fft2c(hf["Original_Data"][:, dataslice])

        # normalized kspace with normalization constant from image:
        if self.normalize == "abs_image":
            abs_simulated_image = np.amax(abs(simulated_image)) + 1e-9
            kspace = fft2c(simulated_image) / abs_simulated_image
        if self.normalize == "line_wise":
            kspace = fft2c(simulated_image)
            if self.subst_with_orig:
                cond = np.rollaxis(np.tile(target_mask, (112, 12, 1)), 0, 3)
                kspace[cond >= self.subst_with_orig] = original_kspace[cond >= self.subst_with_orig]
                if not self.soft_mask:
                    target_mask[target_mask < self.subst_with_orig] = 0
                    target_mask[target_mask >= self.subst_with_orig] = 1
                else:
                    target_mask[target_mask >= self.subst_with_orig] = 1

        if self.crop_readout:
            # crop the readout dimension (last dimension) to specified size:
            ind = int((kspace.shape[-1]-self.crop_readout)/2)
            kspace = kspace[..., ind:-ind]

        if self.magn_phase:
            # only implemented for norm_lines at the moment
            if self.normalize == "abs_image":
                print("ERROR in dataloader: magn_phase is only implemented for norm_lines at the moment!")
            # magnitude and phase as separate channels
            if self.normalize == "line_wise":
                norm = np.sqrt(np.sum(abs(kspace)**2, axis=(0, 2)))+1e-9
                kspace_magn_phase = np.zeros((2, *kspace.shape))
                kspace_magn_phase[0] = abs(kspace)/norm[None, :, None]   # all echoes normalised together
                kspace_magn_phase[1] = np.angle(kspace)

                if self.input_2d:
                    kspace_magn_phase = kspace_magn_phase.reshape((2*kspace.shape[0], *kspace[0].shape))

                return torch.as_tensor(kspace_magn_phase), torch.as_tensor(target_mask), \
                       str(filename), dataslice

        else:
            kspace_realv = np.zeros((2, *kspace.shape))
            if self.normalize == "abs_image":
                kspace_realv[0] = np.real(kspace)
                kspace_realv[1] = np.imag(kspace)
            if self.normalize == "line_wise":
                # rescale each line with norm of the line (all echoes normalised together)
                norm = np.sqrt(np.sum(abs(kspace) ** 2, axis=(0, 2))) + 1e-9
                kspace = kspace / norm[None, :, None]
                kspace_realv[0] = np.real(kspace)
                kspace_realv[1] = np.imag(kspace)

                if self.input_2d:
                    kspace_realv = kspace_realv.reshape((2*kspace.shape[0], *kspace[0].shape))


            return torch.as_tensor(kspace_realv), torch.as_tensor(target_mask), \
                   str(filename), dataslice


class T2starLoader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        akeys = args.keys()
        self.batch_size = args['batch_size'] if 'batch_size' in akeys else 8
        self.only_brainmask_slices = args['only_brainmask_slices'] if 'only_brainmask_slices' in akeys else False
        self.bm_thr = args['bm_thr'] if 'bm_thr' in akeys else 0.1
        self.normalize = args['normalize'] if 'normalize' in akeys else "abs_image"
        self.soft_mask = args['soft_mask'] if 'soft_mask' in akeys else False
        self.subst_with_orig = args['subst_with_orig'] if 'subst_with_orig' in akeys else None
        self.crop_readout = args['crop_readout'] if 'crop_readout' in akeys else False
        self.overfit_one_sample = args['overfit_one_sample'] if 'overfit_one_sample' in akeys else False
        self.magn_phase = args['magn_phase'] if 'magn_phase' in akeys else False
        self.input_2d = args['input_2d'] if 'input_2d' in akeys else False
        self.drop_last = False if self.overfit_one_sample else True
        self.num_workers = args['num_workers'] if 'num_workers' in akeys else 2
        # if 'train_data_path' in akeys:
        #     self.train_data_path = args['train_data_path']
        # else:
        #     logging.info('No training data path specified.')
        # if 'val_data_path' in akeys:
        #     self.val_data_path = args['val_data_path']
        # else:
        #     logging.info('No validation data path specified.')
        # if 'test_data_path' in akeys:
        #     self.test_data_path = args['test_data_path']
        # else:
        #     logging.info('No test data path specified.')
        self.data_dir = args['data_dir'] if 'data_dir' in akeys else None
        assert type(self.data_dir) is dict, 'DefaultDataset::init():  data_dir variable should be a dictionary'


    def train_dataloader(self):
        """Loads a batch of training data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files.
        """
        trainset = T2starDataset(self.data_dir['train'],    #path=self.train_data_path,
                                 only_bm_slices=self.only_brainmask_slices,
                                 bm_thr=self.bm_thr,
                                 normalize=self.normalize,
                                 soft_mask=self.soft_mask,
                                 subst_with_orig=self.subst_with_orig,
                                 magn_phase=self.magn_phase,
                                 input_2d=self.input_2d,
                                 crop_readout=self.crop_readout,
                                 overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the train dataset: {trainset.__len__()}.")

        dataloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        """Loads a batch of validation data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files.
        """
        valset = T2starDataset(self.data_dir['val'],    #self.val_data_path,
                               only_bm_slices=self.only_brainmask_slices,
                               bm_thr=self.bm_thr,
                               normalize=self.normalize,
                               soft_mask=self.soft_mask,
                               subst_with_orig=self.subst_with_orig,
                               magn_phase=self.magn_phase,
                               input_2d=self.input_2d,
                               crop_readout=self.crop_readout,
                               overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the validation dataset: {valset.__len__()}.")

        dataloader = DataLoader(
            valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.drop_last,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        """Loads a batch of testing data consisting of kspace data, target
        mask, filenames and slice indices of the associated h5 files. For the
        test data loader 'drop_last' is enabled, which means that no data will
        be loaded if the batch size is larger than the size of the test set.
        """
        testset = T2starDataset(self.data_dir['test'],    #self.test_data_path,
                                only_bm_slices=self.only_brainmask_slices,
                                bm_thr=self.bm_thr,
                                normalize=self.normalize,
                                soft_mask=self.soft_mask,
                                subst_with_orig=self.subst_with_orig,
                                magn_phase=self.magn_phase,
                                input_2d=self.input_2d,
                                crop_readout=self.crop_readout,
                                overfit_one_sample=self.overfit_one_sample)
        logging.info(f"Size of the test dataset: {testset.__len__()}.")

        if testset.__len__() < self.batch_size:
            logging.info('The batch size ({}) is larger than the size of the test set({})! Since the dataloader has '
                         'drop_last enabled, no data will be loaded!'.format(self.batch_size, testset.__len__()))

        dataloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader


class T2StarDataSample(NamedTuple):
    """Generate named tuples consisting of filename and slice index"""
    fname: pathlib.Path
    slice_ind: int


def fft2c(x, shape=None, dim=(-2, -1)):
    """Centered Fourier transform"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=dim), axes=dim, norm='ortho', s=shape), axes=dim)


def load_all_echoes(in_folder, descr, nr_pe_steps=92, nr_echoes=12, offset=2047):
    """Load all echoes for one acquisition into one array as complex dataset

    :param in_folder: input folder
    :param descr: string describing the files to look for
    :param nr_pe_steps: number of phase encoding steps to be simulated.
    :param nr_echoes: number of echoes that are expected. The default is 12.
    :param offset: offset of scanner for saved intensity values. Default for Philips is 2047.
    :return: a complex array of shape [nr_echoes, nr_slices, n_y, n_x]
    :return: an array of shape [4, 4] containing the affine transform for saving as nifti. For saving, please
             remember to apply np.rollaxis of the images to get the shape [n_y, n_x, n_slices]
    """

    files_real = sorted(glob.glob(in_folder+'*real*'+descr))
    files_imag = sorted(glob.glob(in_folder + '*imaginary*' + descr))

    if len(files_real) != nr_echoes:
        print('ERROR: Less than 12 real images found.')
        print(files_real)

    if len(files_imag) != nr_echoes:
        print('ERROR: Less than 12 imaginary images found.')
        print(files_imag)

    shape = np.shape(np.rollaxis(nib.load(files_real[0]).get_fdata(), 2, 0))
    dataset = np.zeros((nr_echoes, shape[0], nr_pe_steps, shape[2]), dtype=complex)

    for re, im, i in zip(files_real, files_imag, range(0, nr_echoes)):
        # load the unscaled nifti (intensities need to match exactly with what scanner saved)
        # subtract offset of 2047 since scanner (Philips) shifted the intensities
        tmp = (np.rollaxis(nib.load(files_real[i]).dataobj.get_unscaled(), 2, 0) - offset) + \
              1j * (np.rollaxis(nib.load(files_imag[i]).dataobj.get_unscaled(), 2, 0) - offset)
        dataset[i] = tmp[:, int((np.shape(tmp)[1]-nr_pe_steps)/2):-int((np.shape(tmp)[1]-nr_pe_steps)/2)]

    affine = nib.load((files_real[0])).affine
    header = nib.load(files_real[0]).header

    return dataset, affine, header
