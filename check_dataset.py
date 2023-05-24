import nibabel as nib
import os
from PIL import Image
import pandas as pd
import numpy as np
df=pd.DataFrame()
listt=[]
path="D:\COMPAI\iml-dl\data\\fastMRI\\brain_mid_png"
path2="D:\COMPAI\iml-dl\data\CAPS_IXI\ixi_normal_train_last.csv"
df = pd.read_csv(path2)
count=0
a=df['filename'].tolist()
for i in range(len(a)):
    c=a[i].find('png')
    a[i]=a[i][c+4:]
for root, dirs, files in os.walk(path):
    for i in range(len(a)):
        if a[i] not in files and a[i][0]!='s':
            print(a[i])
            count+=1
        print(a[i])    
    break        
print(count)