import numpy as np
import pandas as pd
import os, sys, shutil, glob

data_dir = 'C://Users\\VelocityUser\\Documents\\02_22_data'

target_data = []
for files in glob.glob(os.path.join(data_dir, '*.bbx')):
    target_data.append(pd.read_csv(files, header=0, names=["class_id", "class_name", "x", "y", "width", "height"]))

target_data = pd.concat(target_data, axis = 0, ignore_index= True)
# # Visualize dataset
print(target_data["class_name"].value_counts())
# print('\n')
