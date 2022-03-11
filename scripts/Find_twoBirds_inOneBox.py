import pandas as pd
import os

"""
1. generate csv file
2. find bounding box of two birds overlap
    1) print the file name
    2) print the name of two birds
    3) print the index of two birds
        eg:
        DJI_20210520122506_0105
        Laughing Gull Adult
        Laughing Gull Adult
        11
        12
        In DJI_20210520122506_0105.bbx file, bird species and indexes

Makesure:
1. data_dir only have .bbx file
2. the data_dir should have '/' at the end
3. the cvs files are in different directions (python file direction) 
"""
data_dir = 'C:/Users/karen/Downloads/SS22_03/'
# data_dir = 'C:/Users/karen/Downloads/annotation_1017/'
dirs = [os.listdir(data_dir)]
# data_dir: the direction of the folder
# dirs: file name in the folder
"""
• species id: unique species id in integer.
• species label: species label in words.
• x: Smallest x-axis coordinate of the bounding box; used to identify location in image
• y: Smallest y-axis coordinate of the bounding box; used to identify location in image
• width: width of a bounding box.
• height: height of a bounding box.
"""
# target_file: get the direction of files
# filename: the name of the bbx file
target_file = []
filename = []
for d in dirs[0]:
    file_dirs = os.path.join(data_dir, d)
    filename.append(d[:-4])
    target_file.append(file_dirs)
print(filename)

"""
for t in target_file:
    df=pd.read_csv(t)
    print(df)
Output Example:
  AI Class          Species Desc     X     Y  Width  Height
0    BRPEA   Brown Pelican Adult  8096  3210     96     214
1    LAGUA   Laughing Gull Adult  1676   434     77      77
2    LAGUA   Laughing Gull Adult  7650  4631     58      81
3    LAGUA   Laughing Gull Adult  7738  4910     48      94
4    LAGUA   Laughing Gull Adult  7652  5077     57      80
5    LAGUF  Laughing Gull Flying  7086  3255    261     123
"""

# df: type:dataframe, read what in the file
# creat cvs file
# now the data is readable
for i in range(len(target_file)):
    df = pd.read_csv(target_file[i])
    df.to_csv(filename[i] + '.csv')

"""
box:
rightTop            leftTop
    ↓                   ↓
    $-------------------$
    |                   |
    |                   |
    |                   |
    |                   |
    *-------------------$
    ↑                   ↑
  (x,y)             leftBottom
  rightBottom
"""
print('bounding box with two birds')
num = 0
for i in range(len(target_file)):
    df = pd.read_csv(target_file[i])
    BirdSpe = []
    rightBottom_x = []
    rightBottom_y = []
    rightTop_x = []
    rightTop_y = []
    leftTop_x = []
    leftTop_y = []
    leftBottom_x = []
    leftBottom_y = []
    for j in range(len(df)):
        val4 = df.values[j][2:6]
        rightBottom_x.append(val4[0])
        rightBottom_y.append(val4[1])
        rightTop_x.append(val4[0])
        rightTop_y.append(val4[1] + val4[3])

        leftTop_x.append(val4[0] + val4[2])
        leftTop_y.append(val4[1] + val4[3])
        leftBottom_x.append(val4[0] + val4[2])
        leftBottom_y.append(val4[1])

        valName = df.values[j][1]
        BirdSpe.append(valName)

    for j in range(len(df)):
        for k in range(j, len(df)):
            # test rightBottom - point - leftBottom
            # test rightBottom - point - rightTop

            if (((rightBottom_x[j] < rightBottom_x[k] and rightBottom_x[k] < leftBottom_x[j])
                 and (rightBottom_y[j] < rightBottom_y[k] and rightBottom_y[k] < rightTop_y[j]))
                    or ((rightTop_x[j] < rightTop_x[k] and rightTop_x[k] < leftBottom_x[j])
                        and (rightBottom_y[j] < rightTop_y[k] and rightTop_y[k] < rightTop_y[j]))
                    or ((rightBottom_x[j] < leftTop_x[k] and leftTop_x[k] < leftBottom_x[j])
                        and (rightBottom_y[j] < leftTop_y[k] and leftTop_y[k] < rightTop_y[j]))
                    or ((rightBottom_x[j] < leftBottom_x[k] and leftBottom_x[k] < leftBottom_x[j])
                        and (rightBottom_y[j] < leftBottom_y[k] and leftBottom_y[k] < rightTop_y[j]))):
                """if BirdSpe[j]!=BirdSpe[k]:
                    print(filename[i])
                    print(BirdSpe[j])
                    print(BirdSpe[k])

                    print(j)
                    print(k)
                    print('------------')
DJI_20210520121336_0691
Royal Tern
Sandwich Tern
31
47
                """

                print(filename[i])
                print(BirdSpe[j])
                print(BirdSpe[k])
                num = num + 1
                print(j)
                print(k)
                print('------------')

print(num)
"""
img_dir = 'C:/Users/karen/Downloads/image/DJI_20210520120625_0436.jpg'
#I_dirs = [os.listdir(img_dir)]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
SameBird = mpimg.imread(img_dir)"""
"""
DJI_20210520120625_0436
Royal Tern
Royal Tern
2
3
"""

csvFile = 'DJI_20210520120625_0436.csv'
show = pd.read_csv(csvFile)
bird1_x = show.values[1][3]
bird1_y = show.values[1][4]
bird1_w = show.values[1][5]
bird1_h = show.values[1][6]
img_dir = 'C:/Users/karen/Downloads/image/DJI_20210520120625_0436.jpg'
from PIL import Image
import matplotlib.pyplot as plt

SameBird = Image.open(img_dir)

CropBird1 = SameBird.crop((bird1_x, bird1_y, bird1_x + bird1_w, bird1_y + bird1_h))

"""
plt.figure('same birds')
plt.imshow(CropBird)
plt.axis('off')
plt.show()
"""

# next image

csvFile = 'DJI_20210520121336_0691.csv'
show = pd.read_csv(csvFile)
bird2_x = show.values[47][3]
bird2_y = show.values[47][4]
bird2_w = show.values[47][5]
bird2_h = show.values[47][6]
img_dir = 'C:/Users/karen/Downloads/image/DJI_20210520121336_0691.jpg'

SameBird = Image.open(img_dir)

CropBird2 = SameBird.crop((bird2_x, bird2_y, bird2_x + bird2_w, bird2_y + bird2_h))

"""
plt.imshow(CropBird)
plt.axis('off')
plt.show()
"""

plt.figure('two birds')

plt.subplot(121)
plt.title('same birds', fontsize=15)
plt.axis('off')
plt.imshow(CropBird1)

plt.subplot(122)
plt.title('different birds', fontsize=15)
plt.axis('off')
plt.imshow(CropBird2)
plt.savefig('overlapping birds')

plt.show()
