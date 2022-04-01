# #coding = utf-8
# """
# according to the result of training, some bird species cannot be detected
# 1. find the size of the bounding box of each species
# 2. figuring out why some species has higher accuracy and some cannot be detected
# This file based on the dataset uploaded on Feb 20
# Bird species:
# AI Class                Desc            Total Annotations
# MTRNA           Mixed Tern Adult             8794
# LAGUA          Laughing Gull Adult           2694
# BRPEA         Brown Pelican Adult            309
# GBHEA       Great Blue Heron Adult           214
# TRHEA       Tricolored Heron Adult           212
# ROSPA       Roseate Spoonbill Adult          52
# WHIBA           White Ibis Adult             22
# BCNHA   Black-Crowned Night Heron Adult      21
# REEGA       Reddish Egret Adult              15
# SNEGA           Snowy Egret                  11
# LAGUA       Laughing Gull Flying              3
# REEGWMA   White Morph Reddish Egret Adult     2
# """
# import pandas as pd
# import os
# from numpy import *
# import numpy as np
# """
# • species id: unique species id in integer.
# • species label: species label in words.
# • x: Smallest x-axis coordinate of the bounding box; used to identify location in image
# • y: Smallest y-axis coordinate of the bounding box; used to identify location in image
# • width: width of a bounding box.
# • height: height of a bounding box.
# """
# # **************************************************************************
# #                   size of bounding box
# # **************************************************************************
#
# def dataLoader():
#     # data_dir = 'C:/Users/karen/Downloads/SS22_03/'
#
#     data_dir = 'C:/Users/VelocityUser/Documents/0222_data/'
#     dirs = [os.listdir(data_dir)]
#
#     # target_file: get the direction of files
#     # filename: the name of the bbx file
#     target_file = []
#     filename = []
#     for d in dirs[0]:
#         file_dirs = os.path.join(data_dir, d[:-4])
#         filename.append(d[:-4])
#         if d[:-4] != '20220220 - Five Classes':
#             target_file.append(file_dirs+'.bbx')
#
#     return target_file
#
# def checkoneSpecies(target_file, species_name):
#     width = []
#     height = []
#     for i in range(len(target_file)):
#
#         df = pd.read_csv(target_file[i])
#         for j in range(len(df)):
#             val1 = df.values[j][1]
#
#             if val1 == species_name:
#                 val2 = df.values[j][4:6]
#                 width.append(val2[0])
#                 height.append(val2[1])
#
#     print('min width of ', species_name, ' is ', min(width))
#     print('min height of ', species_name, ' is ', min(height))
#     print('max width of ', species_name, ' is ', max(width))
#     print('max height of ', species_name, ' is ', max(height))
#     print('mean width of ', species_name, ' is ', mean(width))
#     print('mean height of ', species_name, ' is ', mean(height))
#     print('std of the width of ', species_name, ' is ', np.std(width,ddof=1))
#     print('std of the height of ', species_name, ' is ', np.std(height,ddof=1))
#
# target_file = dataLoader()
#
# BirdSpecies = ['Mixed Tern Adult','Laughing Gull Adult','Brown Pelican Adult',
#                'Great Blue Heron Adult','Tricolored Heron Adult','Roseate Spoonbill Adult',
#                'White Ibis Adult','Black-Crowned Night Heron Adult','Reddish Egret Adult',
#                'Snowy Egret','Laughing Gull Flying','White Morph Reddish Egret Adult']
#
# for Bird in BirdSpecies:
#     checkoneSpecies(target_file, Bird)
#     print('---------------------')
#
#
# # **************************************************************************
# #                   plot
# # **************************************************************************
#
#
# import glob, os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io
# import pandas as pd
# import seaborn as sns
#
# """
# data exploration for the dataset last semester
# the "annotation_1017" is a folder with only annotations
# and "SS22_02" is a folder with both annotations and images (same name)
# """
#
#
# def draw_bounding_boxes(img, data, legend=True):
#     """
#     Function to draw bounding boxes onto image
#     INPUTS:
#         img -- <numpy.ndarray> input image for bounding boxes to be placed on
#         data -- <pandas.DataFrame> dataframe containing columns for object class and
#                 bounding box parameters (x, y, width, height)
#         legend -- <boolean> toggle to place legend on plot
#     OUTPUTS:
#         output -- <numpy.ndarray> edited input image with bounding boxes
#     """
#     ######## Function doesn't seem to return any output, just plots
#
#     from matplotlib.patches import Rectangle, Patch
#
#     classes = pd.unique(data["class_name"])
#
#     # colormap
#     cmap = plt.cm.get_cmap("jet")
#     color_ls = np.linspace(0, 1, num=classes.size)
#
#     # draw bounding boxes
#     fig, ax = plt.subplots(figsize=[6, 4], dpi=100)
#     ax.imshow(img)
#     # ax.imshow(np.fliplr(np.flipud(img)))
#     for i in range(data.shape[0]):
#         # for i in range(1):
#         #display(data)
#         ########## how does this work if classes is a list?
#         class_num = np.squeeze(np.argwhere(data["class_name"][i] == classes))
#         rect = Rectangle((data["x"][i], data["y"][i]), data["width"][i], data["height"][i],
#                          edgecolor=cmap(color_ls[class_num])[:3],
#                          linewidth=1, facecolor='none')
#         ax.add_patch(rect)
#
#     ax.set_title("Bounding boxes")
#     # legend
#     if legend:
#         legend_elements = [Patch(facecolor='none', edgecolor=cmap(color_ls[i])[:3], label=c) for i, c in
#                            enumerate(classes)]
#         ax.legend(handles=legend_elements, loc='upper right')
#
#     plt.show()
#
#
# def dataLoader():
#     """
#     Function to load data in for data exploration
#     INPUTS:
#
#     OUTPUTS:
#         target_data -- <pd.dataframe> dataframe with bounding boxes processed and aggregated
#     """
#     #data_dir = 'data/Annotations 20210912/'
#
#     data_dir = 'C:/Users/VelocityUser/Documents/0224_data/'
#     #data_dir = 'C:/Users/karen/Downloads/SS22_03/'
#     # Load CSV files
#     #dir_exceptions = ["Example"]
#     dirs = [os.listdir(data_dir)]
#     #dirs = [d for d in os.listdir(data_dir)
#     #        if d not in dir_exceptions]
#     target_data = []
#     for d in dirs:
#         for f in glob.glob(os.path.join(data_dir, '*.bbx')):
#             df = pd.read_csv(f, header=0, names=["class_id", "class_name", "x", "y", "width", "height"])
#             df['file'] = str(f)  ######### broadcasts the string to all entries?
#             target_data.append(df)
#     target_data = pd.concat(target_data, axis=0, ignore_index=True)
#
#     return target_data
#
#
# def birdCounts(target_data):
#     """
#     Function to display bird counts graphic
#     INPUTS:
#         target_data -- <pd.dataframe> dataframe with bounding boxes processed and aggregated
#     OUTPUTS:
#     """
#     ########## For each class, compute the 'mean' of 'width' and the 'count' and 'mean' of 'height'
#     bounding_box_data = target_data.groupby(['class_name']).agg({'width': ['mean'], 'height': ['mean', 'count']})
#     ########## This average area calculation probably incorrect
#     bounding_box_data[('area', 'mean')] = bounding_box_data[('width', 'mean')] * bounding_box_data[('height', 'mean')]
#
#     ### Bird Counts
#     target_counts = target_data["class_name"].value_counts().sort_values(ascending=False)
#     plt.figure(figsize=(13, 7))
#     print(np.log(target_counts.values))
#     ########## The barplot has bars that are horizontal, I believe
#     ########## target_counts.index is in iterable of the names?
#     ########## target_counts.index.values is to access the names?
#     ########## target_counts.values are the count values?
#     ax = sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
#     ax.set_xlim(right=10000)
#     # Set tick font size
#     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#         label.set_fontsize(16)
#     for i, v in enumerate(target_counts):
#         ax.text(v + 3, i + 0.1, str(v), color='black', fontweight='bold', size=16)
#     plt.xlabel('Number of Birds', fontsize=16)
#     plt.tight_layout()
#     plt.savefig('bird_count.png', dpi=600)
#     plt.show()
#
#
# def birdPerPhoto(target_data):
#     """
#     Function to display bird per photo graphic
#     INPUTS:
#         target_data -- <pd.dataframe> dataframe with bounding boxes processed and aggregated
#     OUTPUTS:
#     """
#     ######### Produces df where for each file (image), the number of observations (birds) is counted
#     by_file = target_data.groupby(["file"]).count()["class_id"]
#     a = list(by_file)
#     print(pd.Series(a).describe())
#     print(a)
#     print(len(set(a)))
#     print(set(a))
#     ok = {"1 to 100": 0, "101 to 200": 0, "201 to 300": 0, "301 to 400": 0, "401 to 500": 0, "501 to 600": 0, "> 601": 0}
#     new_a = []
#     for item in a:
#         if item <= 100:
#             new_a.append("1 to 100")  ########## I'm guessing it doesn't append if it already exists
#             ok["1 to 100"] += 1
#         elif item <= 200:
#             new_a.append("101 to 200")
#             ok["201 to 300"] += 1
#         elif item <= 300:
#             new_a.append("201 to 300")
#             ok["201 to 300"] += 1
#         elif item <= 400:
#             new_a.append("301 to 400")
#             ok["301 to 400"] += 1
#         elif item <= 500:
#             new_a.append("401 to 500")
#             ok["401 to 500"] += 1
#         elif item <= 600:
#             new_a.append("501 to 600")
#             ok["501 to 600"] += 1
#         else:
#             new_a.append("> 601")
#             ok["> 601"] += 1
#
#     labels = ok.keys()
#     sizes = ok.values()
#     explode = (0.1, 0.1, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
#
#     fig1, ax1 = plt.subplots()
#     ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     plt.title("Distribution of Birds per Image")
#     plt.tight_layout()
#     plt.savefig('percentLabelledBirdPerImage.png')
#     plt.show()
#
# def birdExamples():
#     """
#     Function to display examples of each bird
#     INPUTS:
#     OUTPUTS:
#     """
#
#     target_data = {'Great Blue Heron Adult': "", 'Turkey Vulture Adult': "", 'Brown Pelican Adult': "",
#                    'White Ibis Adult': "", 'Ring-Billed Gull Adult': "", 'Great Egret Adult': "",
#                    'Roseate Spoonbill Adult': "", 'Black-Crowned Night Heron Adult': "", 'Snowy Egret': "",
#                    'Laughing Gull Adult': "", 'Tricolored Heron Adult': "",
#                    'White Morph Reddish Egret Adult': "",
#                    'Tri-Colored Heron Adult': "", 'Great Egret/White Morph Adult': "", 'Mixed Tern Adult': "",
#                    'Black Crowned Night Heron Adult': ""}
#
#     counter = 0
#     data_dir = 'C:/Users/VelocityUser/Documents/0224_data/'
#
#     # Load CSV files
#
#     dirs = os.listdir(data_dir)
#
#     # birdSpecies = []
#     for d in dirs:
#         if d[:-4] != '20220220 - Five Classes':
#             if counter == 16:
#                 break
#             for f in glob.glob(os.path.join(data_dir, d[:-4]+'.bbx')):
#                 if counter == 16:    ########### Means an example has been found for all 15 birds
#                     break
#                 df = pd.read_csv(f, header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])
#                 # print(df)
#                 #for j in range(len(df)):
#                     # birdSpecies.append(df.values[j][1])
#                 for key, val in target_data.items():
#
#                     if key in list(df['class_name']) and val == "":
#                         # print(df.loc[df['class_name'] == key].head(1))
#                         #a = f
#                         a = f[:-4] + ".jpg"   ########### a is now the name of the image file
#
#                         df_pass = df.loc[df['class_name'] == key].head(1).reset_index(drop=True)
#                         ########### df_pass is a df for just one instance of the bird type "key"?
#                         df_pass['class'] = df_pass['class_name']
#                         target_data[key] = (a, df_pass)   ######### val is now a tuple
#                         # print(df.loc[df['class_name'] == key].head(1))
#                         print(counter, key, a)
#                         counter += 1
#
#
#     count = 0
#     for k, v in target_data.items():
#
#         # print(v[1]["class_name"][0])
#         image = io.imread(v[0])
#
#
#
#         ########## v[1] is the dataframe
#         # print(v[1]["x"], v[1]["x"] + v[1]["width"], v[1]["y"], v[1]["y"] +  v[1]["height"])
#         ########## This comes from the x and y values in the df being the min x and min y of the box
#         cropImage = image[int(v[1]["y"]):int(v[1]["y"]) + int(v[1]["height"]), int(v[1]["x"]):int(v[1]["x"] + v[1]["width"])]
#     # print(cropImage.shape)
#         count = count+1
#         plt.axis('off')
#         plt.subplot(4,4,count)
#         plt.imshow(cropImage)
#         plt.title(v[1]["class_name"][0], fontsize = 7)
#         # plt.title(v[1]["class_name"][0], fontsize = 7)
#         plt.tight_layout()
#
#
#
#     plt.axis('off')
#     plt.savefig('BirdSpeciesExampleImage.png')
#     plt.show()
#
#
# target_data = dataLoader()
# birdCounts(target_data)
# birdPerPhoto(target_data)
# birdExamples()
# # #
#


"""
This file is to sub-pic images of Roseate Spoonbill
It cannot be detect after doing the training
The size of the bounding box seems have no problems
But it may related to where they appeared most
"""

# import pandas as pd
# import glob
# import matplotlib.pyplot as plt
# import os
# from skimage import io
#
# """
# • species id: unique species id in integer.
# • species label: species label in words.
# • x: Smallest x-axis coordinate of the bounding box; used to identify location in image
# • y: Smallest y-axis coordinate of the bounding box; used to identify location in image
# • width: width of a bounding box.
# • height: height of a bounding box.
# """
#
# def dataLoader():
#     data_dir = 'C:/Users/VelocityUser/Documents/0222_data/'
#
#     dirs = [os.listdir(data_dir)]
#     # dirs = [d for d in os.listdir(data_dir)
#     #        if d not in dir_exceptions]
#     target_file = []
#     for d in dirs:
#         for f in glob.glob(os.path.join(data_dir, '*.bbx')):
#             target_file.append(f)
#     return target_file
#
# def plot_oneBirdS (target_file, species_name):
#     counter = 0
#     image_dirs = []
#     x_min = []
#     x_max = []
#     y_min = []
#     y_max = []
#     for i in range(len(target_file)):
#         df = pd.read_csv(target_file[i])
#         for j in range(len(df)):
#             val1 = df.values[j][1]
#             if (val1 == species_name) and counter<16:
#                 val4 = df.values[j][2:6]
#                 x_min.append(val4[0])
#                 y_min.append(val4[1])
#                 x_max.append(val4[0] + val4[2])
#                 y_max.append(val4[1] + val4[3])
#                 image_dirs.append(target_file[i][:-4]+'.jpg')
#                 counter += 1
#
#     for i in range(len(image_dirs)):
#         image = io.imread(image_dirs[i])
#         print(y_max[i])
#         cropImage = image[y_min[i]:y_max[i],x_min[i]:x_max[i]]
#         print(image_dirs[i])
#         plt.subplot(4, 4, i+1)
#         #plt.axis('off')
#         plt.imshow(cropImage)
#         plt.axis('off')
#
#         #plt.tight_layout()
#
#
#     plt.savefig('GreatBlueHeronAdult.png')
#
#     plt.show()
# #
# target_data = dataLoader()
#
# plot_oneBirdS(target_data,'Great Blue Heron Adult')

#
# import os, glob
# import pandas as pd
#
# # data_dir = 'C:/Users/VelocityUser/Documents/Audubon_F21/0224_data_change_anno/20220210 - LBNI 10k-6-1.bbx'
# # df = pd.read_csv(data_dir)
# # print(df)
# # for i in range(len(df)):
# #     if 'Nest' in df['Species Desc'][i]:
# #         df = df.drop(i)
# # print(df)
#
# data_dir = 'C:/Users/VelocityUser/Documents/Audubon_F21/augment_0224_2/'
#
# dirs = [os.listdir(data_dir)]
#
# target_file = []
# filename = []
# for d in dirs[0]:
#     print(d)
#     if d[-4:] == '.csv':
#         file_dirs = os.path.join(data_dir, d)
#
#         filename.append(d[:-4])
#         target_file.append(file_dirs)
#
# target_data = []
# for i in range(len(target_file)):
#     df = pd.read_csv(target_file[i])
#     df['desc'] = df['desc'].apply(lambda x: 'fly' if ('Flying' in x)or('Wings Spread' in x)or('Flight' in x) else x)
#
#     for j in range(len(df)):
#         if ('Nest' in df['desc'][j]) or ('Egg' in df['desc'][j]) or ('Other Bird' in df['desc'][j]):
#
#             df = df.drop(j)
#     # print(df)
#     # df.drop(index=(df[['Egg' in x for x in df['Species Desc']]]),axis = 0)
#     df.to_csv(data_dir+filename[i] + '.csv')
#
# target_data = []
# for f in glob.glob(os.path.join(data_dir,'*.csv')):
#   target_data.append(pd.read_csv(f, header=0,
#                               names = ["class_id", "class_name", "x", "y", "width", "height"]) )
# target_data = pd.concat(target_data, axis=0, ignore_index=True)
#
# print('\n Bird Species Distribution')
# print(target_data["class_name"].value_counts())
# print('\n')


import os
import pandas as pd


data_dir = 'C:/Users/VelocityUser/Documents/Audubon_F21/augment_0224_2/'
def dropNestAnnotation(data_dir):
dirs = [os.listdir(data_dir)]

target_file = []
filename = []
for d in dirs[0]:
    print(d)
    if d[-4:] == '.csv':
        file_dirs = os.path.join(data_dir, d)

        filename.append(d[:-4])
        target_file.append(file_dirs)

target_data = []
for i in range(len(target_file)):
    df = pd.read_csv(target_file[i])
    df['desc'] = df['desc'].apply(lambda x: 'fly' if ('Flying' in x)or('Wings Spread' in x)or('Flight' in x) else x)

    for j in range(len(df)):
        if ('Nest' in df['desc'][j]) or ('Egg' in df['desc'][j]) or ('Other Bird' in df['desc'][j]):

            df = df.drop(j)
    # print(df)
    # df.drop(index=(df[['Egg' in x for x in df['Species Desc']]]),axis = 0)
    df.to_csv(data_dir+filename[i] + '.csv')