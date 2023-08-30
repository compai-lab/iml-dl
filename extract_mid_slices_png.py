import nibabel as nib
import os
from PIL import Image
import pandas as pd
import numpy as np
df=pd.DataFrame()
listt=[]
path="/home/yigit/iml-dl/data/ADNI/AD_Siem_3T_WW"
path2="/home/yigit/iml-dl/data/ADNI/png"
for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith(("_skullfree_Warped_reshaped.nii.gz")):
            img = nib.load(root+'/'+name)
            data = img.get_fdata()
            data=np.squeeze(data[:,:,round(data.shape[2]/2)])
            data=Image.fromarray(data)
            data = data.convert("L")
            data.save(path2+"/"+name[:-7]+".png")
            listt.append("./data/ADNI/png/"+name[:-7]+".png")

df = pd.DataFrame(listt)
#df.to_csv("cn_siem_3T_ww_train_png.csv", sep='\t', encoding='utf-8',index=False)