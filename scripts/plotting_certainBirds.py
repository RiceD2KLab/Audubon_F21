"""
This file is to sub-pic images of Roseate Spoonbill
It cannot be detect after doing the training
The size of the bounding box seems have no problems
But it may related to where they appeared most
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
from skimage import io

"""
• species id: unique species id in integer.
• species label: species label in words.
• x: Smallest x-axis coordinate of the bounding box; used to identify location in image
• y: Smallest y-axis coordinate of the bounding box; used to identify location in image
• width: width of a bounding box.
• height: height of a bounding box.
"""

def dataLoader():
    data_dir = 'C:/Users/karen/Downloads/SS22_02/'
    # data_dir = 'C:/Users/karen/Downloads/annotation_1017/'
    dirs = [os.listdir(data_dir)]
    # dirs = [d for d in os.listdir(data_dir)
    #        if d not in dir_exceptions]
    target_file = []
    for d in dirs:
        for f in glob.glob(os.path.join(data_dir, '*.bbx')):
            target_file.append(f)
    return target_file

def plot_oneBirdS (target_file, species_name):
    counter = 0
    image_dirs = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    for i in range(len(target_file)):
        df = pd.read_csv(target_file[i])
        for j in range(len(df)):
            val1 = df.values[j][1]
            if (val1 == species_name) and counter<16:
                val4 = df.values[j][2:6]
                x_min.append(val4[0])
                y_min.append(val4[1])
                x_max.append(val4[0] + val4[2])
                y_max.append(val4[1] + val4[3])
                image_dirs.append(target_file[i][:-4]+'.jpg')
                counter += 1

    for i in range(len(image_dirs)):
        image = io.imread(image_dirs[i])
        print(y_max[i])
        cropImage = image[y_min[i]:y_max[i],x_min[i]:x_max[i]]
        print(image_dirs[i])
        plt.subplot(4, 4, i+1)
        #plt.axis('off')
        plt.imshow(cropImage)
        #plt.title(species_name, fontsize=7)
        #plt.tight_layout()

    #plt.axis('off')
    #plt.savefig('BirdSpeciesExampleImage.png')
    plt.show()
#
target_data = dataLoader()

plot_oneBirdS(target_data,'Royal Tern')

