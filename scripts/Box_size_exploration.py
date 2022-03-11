"""
according to the result of training, some bird species cannot be detected
1. find the size of the bounding box of each species
2. figuring out why some species has higher accuracy and some cannot be detected

This file based on the dataset uploaded on Feb 20
Bird species:

AI Class                Desc            Total Annotations
MTRNA           Mixed Tern Adult             8794
LAGUA          Laughing Gull Adult           2694
BRPEA         Brown Pelican Adult            309
GBHEA       Great Blue Heron Adult           214
TRHEA       Tricolored Heron Adult           212
ROSPA       Roseate Spoonbill Adult          52
WHIBA           White Ibis Adult             22
BCNHA   Black-Crowned Night Heron Adult      21
REEGA       Reddish Egret Adult              15
SNEGA           Snowy Egret                  11
LAGUA       Laughing Gull Flying              3
REEGWMA   White Morph Reddish Egret Adult     2


"""
import pandas as pd
import os
from numpy import *
import numpy as np
"""
• species id: unique species id in integer.
• species label: species label in words.
• x: Smallest x-axis coordinate of the bounding box; used to identify location in image
• y: Smallest y-axis coordinate of the bounding box; used to identify location in image
• width: width of a bounding box.
• height: height of a bounding box.
"""

def dataLoader():
    data_dir = 'C:/Users/karen/Downloads/SS22_03/'
    # data_dir = 'C:/Users/karen/Downloads/annotation_1017/'
    dirs = [os.listdir(data_dir)]


    # target_file: get the direction of files
    # filename: the name of the bbx file
    target_file = []
    filename = []
    for d in dirs[0]:
        file_dirs = os.path.join(data_dir, d)
        filename.append(d[:-4])
        target_file.append(file_dirs)
    return target_file

def checkoneSpecies(target_file, species_name):
    width = []
    height = []
    for i in range(len(target_file)):
        df = pd.read_csv(target_file[i])
        print(df)
        for j in range(len(df)):
            val1 = df.values[j][1]
            if val1 == species_name:
                val2 = df.values[j][4:6]
                width.append(val2[0])
                height.append(val2[1])
    print('min width of ', species_name, ' is ', min(width))
    print('min height of ', species_name, ' is ', min(height))
    print('max width of ', species_name, ' is ', max(width))
    print('max height of ', species_name, ' is ', max(height))
    print('mean width of ', species_name, ' is ', mean(width))
    print('mean height of ', species_name, ' is ', mean(height))
    print('std of the width of ', species_name, ' is ', np.std(width,ddof=1))
    print('std of the height of ', species_name, ' is ', np.std(height,ddof=1))

target_file = dataLoader()

"""
BirdSpecies = ['Mixed Tern Adult','Laughing Gull Adult','Brown Pelican Adult',
               'Great Blue Heron Adult','Tricolored Heron Adult','Roseate Spoonbill Adult',
               'White Ibis Adult','Black-Crowned Night Heron Adult','Reddish Egret Adult',
               'Snowy Egret','Laughing Gull Flying','White Morph Reddish Egret Adult']"""
BirdSpecies = ['Sandwich Tern',
                   'Royal Tern']
for Bird in BirdSpecies:
    checkoneSpecies(target_file, Bird)
    print('---------------------')
