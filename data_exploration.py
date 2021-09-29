import glob, os
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io 
import pandas as pd
import seaborn as sns
%matplotlib inline

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
  from matplotlib.patches import Rectangle, Patch

  classes = pd.unique(data["class_name"])
  
  # colormap 
  cmap = plt.cm.get_cmap("jet")
  color_ls = np.linspace(0,1,num=classes.size)
  
  # draw bounding boxes
  fig, ax = plt.subplots(figsize=[6,4], dpi=100)
  ax.imshow(img)
  # ax.imshow(np.fliplr(np.flipud(img)))
  for i in range(data.shape[0]): 
  # for i in range(1):
    display(data)
    class_num = np.squeeze(np.argwhere(data["class_name"][i]==classes))
    rect = Rectangle((data["x"][i],data["y"][i]),data["width"][i],data["height"][i],
                                 edgecolor=cmap(color_ls[class_num])[:3], 
                                 linewidth=1, facecolor='none')
    ax.add_patch(rect)

  ax.set_title("Bounding boxes")  
  # legend
  if legend: 
    legend_elements = [Patch(facecolor='none',edgecolor=cmap(color_ls[i])[:3],label=c) for i,c in enumerate(classes)]
    ax.legend(handles=legend_elements, loc='upper right')
  
  plt.show()

def dataLoader():
    data_dir = 'data/Annotations 20210912/'

    # Load CSV files 
    dir_exceptions = ["Example"]
    dirs = [d for d in os.listdir(data_dir) 
            if d not in dir_exceptions]
    target_data = []
    for d in dirs: 
        for f in glob.glob(os.path.join(data_dir,d,'*.csv')): 
            df = pd.read_csv(f, header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])
            df['file'] = str(f)
            target_data.append(df)
    target_data = pd.concat(target_data, axis=0, ignore_index=True)
    
    return target_data

def birdCounts(target_data):
    bounding_box_data = target_data.groupby(['class_name']).agg({'width': ['mean'], 'height': ['mean', 'count']})

    bounding_box_data[('area',  'mean')] = bounding_box_data[( 'width',  'mean')] * bounding_box_data[('height',  'mean')]

    ### Bird Counts
    target_counts = target_data["class_name"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(13,7))
    print(np.log(target_counts.values))
    ax = sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
    ax.set_xlim(right = 1815)
    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    for i, v in enumerate(target_counts):
        ax.text(v + 3, i + 0.1, str(v), color='black', fontweight='bold', size = 16)
    plt.xlabel('Number of Birds', fontsize=16)
    plt.tight_layout()
    plt.savefig('bird_count.png', dpi = 600)
    plt.show()

def birdPerPhoto(target_data):
    by_file = target_data.groupby(["file"]).count()["class_id"]
    a = list(by_file)
    print(pd.Series(a).describe())
    print(a)
    print(len(set(a)))
    print(set(a))
    ok = {"1 to 20": 0, "21 to 40": 0, "41 to 60": 0, "61 to 80": 0, "81 to 100": 0, "101 to 120": 0, "> 121": 0}
    new_a = []
    for item in a:
        if item <= 20:
            new_a.append("1 to 20")
            ok["1 to 20"] += 1
        elif item <= 40:
            new_a.append("21 to 40")
            ok["21 to 40"] += 1
        elif item <= 60:
            new_a.append("41 to 60")
            ok["41 to 60"] += 1
        elif item <= 80:
            new_a.append("61 to 80")
            ok["61 to 80"] += 1
        elif item <= 100:
            new_a.append("81 to 100")
            ok["81 to 100"] += 1
        elif item <= 120:
            new_a.append("101 to 120")
            ok["101 to 120"] += 1
        else:
            new_a.append("> 121")
            ok["> 121"] += 1

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
    target_data = {'Avocet': "", 'Black Crowned Night Heron': "", 'Brown Pelican': "", 'Cormorant': "",
 'Great Blue Heron': "", 'Great Egret/White Morph': "", 'Herring/Ringbilled Gull': "",
 'Laughing Gull': "", 'Mixed Tern': "", 'Other/Unknown': "", 'Reddish Egret': "",
 'Roseate Spoonbill': "", 'Trash/Debris': "", 'TriColored Heron': "", 'White Ibis': ""}
    counter = 0
    data_dir = 'data/Annotations 20210912/'

    # Load CSV files 
    dir_exceptions = ["Example"]
    dirs = [d for d in os.listdir(data_dir) 
            if d not in dir_exceptions]
    for d in dirs: 
        if counter == 15:
            break
        for f in glob.glob(os.path.join(data_dir,d,'*.csv')): 
            if counter == 15:
                break
            df = pd.read_csv(f, header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])
            for key, val in target_data.items():
                if val == "" and key in list(df['class_name']):
                    # print(df.loc[df['class_name'] == key].head(1))
                    a = f
                    a = a[:-3] + "JPG"
                    df_pass = df.loc[df['class_name'] == key].head(1).reset_index(drop=True)
                    df_pass['class'] = df_pass['class_name']
                    target_data[key] = (a, df_pass)
                    # print(df.loc[df['class_name'] == key].head(1))
                    print(counter, key, a)
                    counter += 1

    # train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"],axis=1).sum(axis=1)
    # count_perc = np.round(100 * train_labels["number_of_targets"].value_counts() / train_labels.shape[0], 2)
    # plt.figure(figsize=(20,5))
    # sns.barplot(x=count_perc.index.values, y=count_perc.values, palette="Reds")
    # plt.xlabel("Number of targets per image")
    # plt.ylabel("% of train data")

    for k, v in target_data.items():
    #   print(v[1])
        image = io.imread(v[0])
        #print(image.shape)

        # print(v[1]["x"], v[1]["x"] + v[1]["width"], v[1]["y"], v[1]["y"] +  v[1]["height"])
        cropImage = image[int(v[1]["y"]):int(v[1]["y"]) + int(v[1]["height"]), int(v[1]["x"]):int(v[1]["x"] + v[1]["width"])]
    # print(cropImage.shape)
        plt.imshow(cropImage)
        plt.tight_layout()
        # print(v[1]["class_name"][0])

        ### Save Fig
        # plt.savefig(f'../gdrive/MyDrive/DSCI535-Audubon/Presentations & Reports/figures/{v[1]["class_name"][0].replace("/", " ")}.png')
        plt.show()


if __name__ == 'main':
    target_data = dataLoader()
    birdCounts(target_data)
    birdPerPhoto(target_data)
    birdExamples()

