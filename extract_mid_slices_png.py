import nibabel as nib
import os
from PIL import Image
import pandas as pd
import numpy as np
df=pd.DataFrame()
listt=[]
path="D:\COMPAI\iml-dl\data\ADNI_AD"
path2="D:\COMPAI\iml-dl\data\ADNI_AD\png\\"
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith((".nii")):
            img = nib.load(root+'\\'+name)
            data = img.get_fdata()
            data=np.squeeze(data[:,:,round(data.shape[2]/2+20)])
            data=Image.fromarray(data)
            data = data.convert("L")
            data.save(path2+name[:-7]+".png")
            listt.append("./data/ADNI_AD/png/"+name[:-7]+".png")

df = pd.DataFrame(listt)
df.to_csv("adni_female_AD_2D.csv", sep='\t', encoding='utf-8',index=False)