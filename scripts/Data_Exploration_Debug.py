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
    data_dir = 'C:/Users/karen/Downloads/annotation_1017/'
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
    ax = sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
    ax.set_xlim(right=9600)
    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    for i, v in enumerate(target_counts):
        ax.text(v + 3, i + 0.1, str(v), color='black', fontweight='bold', size=16)
    plt.xlabel('Number of Birds', fontsize=16)
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
    ok = {"1 to 100": 0, "101 to 200": 0, "201 to 300": 0, "301 to 400": 0, "401 to 500": 0, "501 to 600": 0, "> 601": 0}
    new_a = []
    for item in a:
        if item <= 100:
            new_a.append("1 to 100")  ########## I'm guessing it doesn't append if it already exists
            ok["1 to 100"] += 1
        elif item <= 200:
            new_a.append("101 to 200")
            ok["201 to 300"] += 1
        elif item <= 300:
            new_a.append("201 to 300")
            ok["201 to 300"] += 1
        elif item <= 400:
            new_a.append("301 to 400")
            ok["301 to 400"] += 1
        elif item <= 500:
            new_a.append("401 to 500")
            ok["401 to 500"] += 1
        elif item <= 600:
            new_a.append("501 to 600")
            ok["501 to 600"] += 1
        else:
            new_a.append("> 601")
            ok["> 601"] += 1

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
    target_data = {'Laughing Gull Adult': "", 'Laughing Gull Flying': "", 'Sandwich Tern': "",
                   'Brown Pelican Adult': "", 'Roseate Spoonbill Adult': "", 'Other Bird': "",
                   'Brown Pelican Juvenile': "", 'Brown Pelican - Wings Spread': "",
                   'Royal Tern': "", 'Brown Pelican Chick': "", 'Tri-Colored Heron Adult': "",
                   'Black Crowned Night Heron Adult': "", 'Trash/Debris': "", 'Unsure': "",
                   'Great Egret/White Morph Adult': "", 'Brown Pelican In Flight': ""}
    counter = 0
    data_dir = 'C:/Users/karen/Downloads/SS22_02/'

    # Load CSV files

    dirs = os.listdir(data_dir)

    # birdSpecies = []
    for d in dirs:
        if counter == 16:
            break
        for f in glob.glob(os.path.join(data_dir, d[:-4]+'.bbx')):
            if counter == 16:    ########### Means an example has been found for all 15 birds
                break
            df = pd.read_csv(f, header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])
            print(len(df))
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

    # print(set(birdSpecies))

    count = 0
    for k, v in target_data.items():
    #   print(v[1])
        image = io.imread(v[0])   ########## v[0] is a, the name of the image file
        #print(image.shape)

        ########## v[1] is the dataframe
        # print(v[1]["x"], v[1]["x"] + v[1]["width"], v[1]["y"], v[1]["y"] +  v[1]["height"])
        ########## This comes from the x and y values in the df being the min x and min y of the box
        cropImage = image[int(v[1]["y"]):int(v[1]["y"]) + int(v[1]["height"]), int(v[1]["x"]):int(v[1]["x"] + v[1]["width"])]
    # print(cropImage.shape)
        count = count+1
        plt.axis('off')
        plt.subplot(4,4,count)
        plt.imshow(cropImage)
        plt.title(v[1]["class_name"][0], fontsize = 7)
        plt.tight_layout()


    plt.axis('off')
    plt.savefig('BirdSpeciesExampleImage.png')
    plt.show()


target_data = dataLoader()
birdCounts(target_data)
birdPerPhoto(target_data)
birdExamples()




