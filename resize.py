import nibabel as nib
import os
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from core.DataLoader import DefaultDataset
import torchvision.transforms as transforms
from transforms.preprocessing import *
import torchio

default_t = transforms.Compose([#ReadImage()#, Norm98(),
                                        Pad3D()  # Flip(), #  Slice(),
                                        ,AddChannelIfNeeded(dim=3)
                                       ,Resize3D((128, 128, 128))
                                        # ,AdjustIntensity()
                                        # ,transforms.ToPILImage(), transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1)),
                                        # transforms.RandomHorizontalFlip(0.5),
                                        # transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8,1.2)),
                                        # ,transforms.ToTensor()
                                        ])
listt=[]
path="/home/yigit/iml-dl/data/ADNI/AD_Siem_3T_WW"
for root, dirs, files in os.walk(path):
    for name in files:
        if name.startswith("ADNI") and name.endswith(("_skullfree_Warped.nii.gz")):
            print(name)
            img=nib.load(root+"/"+name)
            img_np=img.get_fdata()
            img_np=default_t(img_np)
            final_img = nib.Nifti1Image(img_np.detach().cpu().numpy().squeeze(), img.affine)
            nib.save(final_img, os.path.join(root,name[:name.find(".nii.gz")]+"_reshaped.nii.gz"))


