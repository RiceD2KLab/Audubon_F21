# THIS IS THE OLD BRANCH USED BY SPRING 2023 TEAM. PLEASE REFER TO THE avialert BRANCH TO SEE OUR FINAL REPOSITORY

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Project description
Houston Audubon uses drone photography to monitor populations of colonial waterbirds in the Texas Gulf Coast. Researchers need to comb through high resolution images and manually count up birds by species class, which can take weeks for even one image. Seeking to automate the annotation process, Houston Audubon partnered with the Data to Knowledge (D2K) lab at Rice University. Student teams have developed an object detection based deep learning model that can automatically detect birds within a drone image and classify their species. This semester, we will continue the project with two main objectives:
  1. Improve the species classification capabilities of the model.
  2. Develop an AI-assisted waterbird annotation tool.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Prerequisites
The following open source packages are used in this project:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `torch`
  - `tqdm`
  - `cv2`
  - `pycocotools`

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Folder structure

  - `utils` || Directory containing helper functions for data processing and exploration
    - `data_processsing.py`
    - `data_vis.py`
    - `crop.py`
  - `audubon_midpoint_demo.ipynb` || Colab notebook containing a demonstration of data processing, data exploration and building a bird-only detection model
  - `README.md`
  - `const.py` || File containing constant parameters, such as column names and hierarchical bird groups
  - `requirements.txt` || List of dependencies
  - `train.py` || File containing helper functions for building a bird-only detection model

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Installation instructions

 ### Clone the repository

  ```linux
  git clone -b SP23 https://github.com/RiceD2KLab/Audubon_F21.git
  ```
 ### Install Pytorch

  <a href="https://pytorch.org/get-started/locally/">Installation instructions here</a> <br>

  Requirements: Linux or macOS with Python ≥ 3.6

  ```linux
  pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/cu102/torch_stable.html
  ```

 ### Install other dependencies

  The Python version is 3.8.16 and users can install the required packages using the following command:

  ```linux
  pip install requirements.txt
  ```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Data science pipeline
Our pipeline is split into three sections: data wrangling, modeling, and model evaluation. Before wrangling, we will split the data into training, validation and testing groups. Overall, we plan to use a similar data science pipeline as past groups. This is especially true for the bird detector, given the model is already highly accurate. This semester we will focus on improving the bird classifier and developing the output (annotation tool).

<img src="imgs/pipeline.png">

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Dataset
Houston Audubon has collected 52 GB of raw images using a DJI M300RTK UAV with a P1 Camera attachment. The images are 8192 X 5460 pixels. For training and testing our model, Houston Audubon has provided us with a 4 GB subset of raw images with annotations for each bird.

Each annotated UAV image has a corresponding CSV file containing bird annotations. Each bird is encapsulated by a bounding box in the image, and each bounding box represents a row in the CSV file. Each row contains the following data about the bird within the bounding box:

  - Unique four letter bird class identifier 
  - Bird class name 
  - Smallest x coordinate in the bounding box (coordinates in terms of pixels)
  - Smallest y coordinate in the bounding box
  - Width of the bounding box
  - Height of the bounding box

In order to access the dataset in Colab, use the following code:

```
!gdown -q 'https://drive.google.com/uc?id=1oK_1Y16dMwcqytFd41UNHxP_O7ju22bo'
!unzip -q './S23-Audubon.zip' && mv S23-Audubon [desired folder in working directory] 
```

The dataset is stored in Google Drive, and must be accessed using the !gdown command. For a demonstration, see the notebook linked below.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Bird detector usage instructions
Currently, we have completed the data processing, data exploration and train/test/split steps of our pipeline. We are working on data wrangling for the bird detection and species classification models. We have created a basic bird detector model, which uses a Faster-RCNN architecture with a ResNet50 backbone network. We still need to optimize the model by using data wrangling techniques and tuning hyperparameters. 

We have created a demonstration notebook containing steps for data preprocessing, data exploration, train/test/split, and training a bird detector. 

Open the [Colab link](https://colab.research.google.com/drive/1fX0P7MlUBt2rgEvnwX7u7j4OnZMwNPG_?usp=sharing) to run the demonstration notebook.
