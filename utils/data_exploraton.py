# #coding = utf-8
# # **************************************************************************
# #                   plot
# # **************************************************************************
#
#
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
import seaborn as sns

"""
data exploration for the dataset last semester
the "annotation_1017" is a folder with only annotations
and "SS22_02" is a folder with both annotations and images (same name)
"""


def draw_bounding_boxes(img, data, legend=True):
    """
    Function to draw bounding boxes onto image
    INPUTS:
        img -- <numpy.ndarray> input image for bounding boxes to be placed on
        data -- <pandas.DataFrame> dataframe containing columns for object class and
                bounding box parameters (x, y, width, height)
        legend -- <boolean> toggle to place legend on plot
    OUTPUTS:
        output -- <numpy.ndarray> edited input image with bounding boxes
    """
    ######## Function doesn't seem to return any output, just plots

    from matplotlib.patches import Rectangle, Patch

    classes = pd.unique(data["class_name"])

    # colormap
    cmap = plt.cm.get_cmap("jet")
    color_ls = np.linspace(0, 1, num=classes.size)

    # draw bounding boxes
    fig, ax = plt.subplots(figsize=[6, 4], dpi=100)
    ax.imshow(img)
    # ax.imshow(np.fliplr(np.flipud(img)))
    for i in range(data.shape[0]):
        # for i in range(1):
        #display(data)
        ########## how does this work if classes is a list?
        class_num = np.squeeze(np.argwhere(data["class_name"][i] == classes))
        rect = Rectangle((data["x"][i], data["y"][i]), data["width"][i], data["height"][i],
                         edgecolor=cmap(color_ls[class_num])[:3],
                         linewidth=1, facecolor='none')
        ax.add_patch(rect)

    ax.set_title("Bounding boxes")
    # legend
    if legend:
        legend_elements = [Patch(facecolor='none', edgecolor=cmap(color_ls[i])[:3], label=c) for i, c in
                           enumerate(classes)]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.show()


def dataLoader():
    """
    Function to load data in for data exploration
    INPUTS:

    OUTPUTS:
        target_data -- <pd.dataframe> dataframe with bounding boxes processed and aggregated
    """
    #data_dir = 'data/Annotations 20210912/'

    data_dir = 'C:/Users/VelocityUser/Documents/D2K TDS A/TDS A-10_com/'
    #data_dir = 'C:/Users/karen/Downloads/SS22_03/'
    # Load CSV files
    #dir_exceptions = ["Example"]
    dirs = [os.listdir(data_dir)]
    #dirs = [d for d in os.listdir(data_dir)
    #        if d not in dir_exceptions]
    target_data = []
    for d in dirs:
        for f in glob.glob(os.path.join(data_dir, '*.bbx')):
            df = pd.read_csv(f, header=0, names=["class_id", "class_name", "x", "y", "width", "height"])
            df['file'] = str(f)  ######### broadcasts the string to all entries?
            target_data.append(df)
    target_data = pd.concat(target_data, axis=0, ignore_index=True)

    return target_data


def birdCounts(target_data):
    """
    Function to display bird counts graphic
    INPUTS:
        target_data -- <pd.dataframe> dataframe with bounding boxes processed and aggregated
    OUTPUTS:
    """
    ########## For each class, compute the 'mean' of 'width' and the 'count' and 'mean' of 'height'
    bounding_box_data = target_data.groupby(['class_name']).agg({'width': ['mean'], 'height': ['mean', 'count']})
    ########## This average area calculation probably incorrect
    bounding_box_data[('area', 'mean')] = bounding_box_data[('width', 'mean')] * bounding_box_data[('height', 'mean')]

    ### Bird Counts
    target_counts = target_data["class_name"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(13, 7))
    print(np.log(target_counts.values))
    ########## The barplot has bars that are horizontal, I believe
    ########## target_counts.index is in iterable of the names?
    ########## target_counts.index.values is to access the names?
    ########## target_counts.values are the count values?
    print(target_counts.values)
    print(target_counts.index.values)

    # color = ['#c1cbd7', '#dfd7d7', '#d8caaf', '#d3d4cc', '#e0cdcf', '#f8ebd8', '#c9c0d3', '#ead0d1', '#96a48b', '#a6a6a8']
    ax = sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index, orient = 'h',
                     palette = "Set2")
    ax.set(xlim=(1, 8700))
    # Set tick font size
    #ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(10)
    for i, v in enumerate(target_counts):
        ax.text(v, i, str(v), color='black', size=12)
    plt.xticks(fontsize=8)
    plt.subplots_adjust(left=0.27)
    # plt.legend(labels=target_counts.index.values.tolist())
    plt.yticks(fontsize=8)
    plt.ylabel('Bird Species', fontsize=12)
    plt.xlabel('Number of Birds', fontsize=12)
    plt.tight_layout()
    plt.savefig('bird_count.png', dpi=600)
    plt.show()


def birdPerPhoto(target_data):
    """
    Function to display bird per photo graphic
    INPUTS:
        target_data -- <pd.dataframe> dataframe with bounding boxes processed and aggregated
    OUTPUTS:
    """
    ######### Produces df where for each file (image), the number of observations (birds) is counted
    by_file = target_data.groupby(["file"]).count()["class_id"]
    a = list(by_file)
    print(pd.Series(a).describe())
    print(a)
    print(len(set(a)))
    print(set(a))
    ok = {"1 to 3": 0, "4 to 6": 0, "7 to 10": 0, "11 to 15": 0, "16 to 25": 0, "26 to 50": 0, "> 50": 0}
    new_a = []
    for item in a:
        if item <= 3:
            new_a.append("1 to 3")  ########## I'm guessing it doesn't append if it already exists
            ok["1 to 3"] += 1
        elif item <= 6:
            new_a.append("4 to 6")
            ok["4 to 6"] += 1
        elif item <= 10:
            new_a.append("7 to 10")
            ok["7 to 10"] += 1
        elif item <= 15:
            new_a.append("11 to 15")
            ok["11 to 15"] += 1
        elif item <= 25:
            new_a.append("16 to 25")
            ok["16 to 25"] += 1
        elif item <= 50:
            new_a.append("26 to 50")
            ok["26 to 50"] += 1
        else:
            new_a.append("> 50")
            ok["> 50"] += 1

    labels = ok.keys()
    sizes = ok.values()
    explode = (0.1, 0.1, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Distribution of Birds per Image")
    plt.tight_layout()
    plt.savefig('percentLabelledBirdPerImage.png')
    plt.show()

def birdExamples():
    """
    Function to display examples of each bird
    INPUTS:
    OUTPUTS:
    """

    target_data = {'Mixed Tern Adult': "", 'Laughing Gull Adult': "",
                   'Black Skimmer Adult': "",'Brown Pelican Adult': "",
                   'Tricolored Heron Adult': "",'Black-Crowned Night Heron Adult': "",
                   'White Ibis Adult': "", 'Reddish Egret Adult': "",
                    'Roseate Spoonbill Adult': "",'Great Blue Heron Adult': ""}

    counter = 0
    data_dir = 'C:/Users/VelocityUser/Documents/D2K TDS A/TDS A-10_com/'

    # Load CSV files

    dirs = os.listdir(data_dir)

    # birdSpecies = []
    for d in dirs:
        if d[:-4] != '20220220 - Five Classes':
            if counter == 16:
                break
            for f in glob.glob(os.path.join(data_dir, d[:-4]+'.bbx')):
                if counter == 16:    ########### Means an example has been found for all 15 birds
                    break
                df = pd.read_csv(f, header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])
                # print(df)
                #for j in range(len(df)):
                    # birdSpecies.append(df.values[j][1])
                for key, val in target_data.items():

                    if key in list(df['class_name']) and val == "":
                        # print(df.loc[df['class_name'] == key].head(1))
                        #a = f
                        a = f[:-4] + ".jpg"   ########### a is now the name of the image file

                        df_pass = df.loc[df['class_name'] == key].head(1).reset_index(drop=True)
                        ########### df_pass is a df for just one instance of the bird type "key"?
                        df_pass['class'] = df_pass['class_name']
                        target_data[key] = (a, df_pass)   ######### val is now a tuple
                        # print(df.loc[df['class_name'] == key].head(1))
                        print(counter, key, a)
                        counter += 1


    count = 0
    for k, v in target_data.items():

        # print(v[1]["class_name"][0])
        image = io.imread(v[0])



        ########## v[1] is the dataframe
        # print(v[1]["x"], v[1]["x"] + v[1]["width"], v[1]["y"], v[1]["y"] +  v[1]["height"])
        ########## This comes from the x and y values in the df being the min x and min y of the box
        cropImage = image[int(v[1]["y"]):int(v[1]["y"]) + int(v[1]["height"]), int(v[1]["x"]):int(v[1]["x"] + v[1]["width"])]
    # print(cropImage.shape)
        count = count+1
        plt.axis('off')

        plt.subplot(4,3,count)
        plt.imshow(cropImage)
        plt.title(v[1]["class_name"][0], fontsize = 8)
        # plt.title(v[1]["class_name"][0], fontsize = 7)
        plt.tight_layout()



    plt.axis('off')
    plt.savefig('BirdSpeciesExampleImage.png')
    plt.show()


target_data = dataLoader()
birdCounts(target_data)
birdPerPhoto(target_data)
birdExamples()
#



# target_data = {'Mixed Tern Adult': "", 'Laughing Gull Adult': "",
#                    'Black Skimmer Adult': "",'Brown Pelican Adult': "",
#                    'Tricolored Heron Adult': "",'Black-Crowned Night Heron Adult': "",
#                    'White Ibis Adult': "", 'Reddish Egret Adult': "",
#                     'Roseate Spoonbill Adult': "",'Great Blue Heron Adult': ""}
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

"""
• species id: unique species id in integer.
• species label: species label in words.
• x: Smallest x-axis coordinate of the bounding box; used to identify location in image
• y: Smallest y-axis coordinate of the bounding box; used to identify location in image
• width: width of a bounding box.
• height: height of a bounding box.
"""
# #
# def dataLoader():
#     data_dir = 'C:/Users/VelocityUser/Documents/D2K TDS A/TDS A-10/Train/'
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



