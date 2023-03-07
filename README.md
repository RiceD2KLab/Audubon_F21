# D2K Team Audubon Spring 2023

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Project description
Houston Audubon uses drone photography to monitor populations of colonial waterbirds in the Texas Gulf Coast. Researchers need to comb through drone images and manually count up birds by species class, which can take weeks for even one high resolution image. Seeking to automate this process, Houston Audubon partnered with the Data to Knowledge (D2K) lab at Rice University. Student teams have developed an object detection based deep learning model that can automatically detect birds within a UAV image and classify their species. This semester, we are continuing the project with two main objecitves:
  1. Improve species classification capabilities of the model.
  2. Develop an AI-assisted waterbird annotation tool.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
  
## Prerequisites
The following open source packages are used in this project:
  - Numpy
  - Pandas
  - Matplotlib
  - PyTorch
  - tqdm

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Folder structure
 
  - utils
    - data_processsing.py
    - data_vis.py
  - README.md
  - const.py
  - requirements.txt
  - train.py

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Installation instructions

 ### Clone the repository

  ```linux
  git clone https://github.com/RiceD2KLab/Audubon_F21.git
  ```
 ### Install Pytorch

  <a href="https://pytorch.org/get-started/locally/">Installation instructions here</a> <br>
  
  Requirements: Linux or macOS with Python ≥ 3.6
  
  ```linux
  pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
  ```
  
 ### Install other dependencies

  The Python version is 3.8.16 and users can install the required packages using the following command:
  
  ```linux
  pip install requirements.txt
  ```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Data Science Pipeline
Our pipeline is split into three sections: data wrangling, modeling, and model evaluation. Before wrangling, we will split the data into training, validation and testing groups. Overall, we plan to use a similar data science pipeline as past groups. This is especially true for the bird detector, given the model is already highly accurate. This semester we will focus on improving the bird classifier and developing the output (annotation tool).

<img src="pipeline.png">

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
  !gdown -q "1hoP1ev8Npj5m0MZWZU7LpjU9c8JYYoFe&confirm=t" 
  !unzip -q './F21-S22-Combined-D2K-Audubon.zip' -d '[desired folder in working directory]' 
```

The dataset is stored in Google Drive, and must be accessed using the !gdown command. 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Bird detector usage instructions
Currently, we have completed the data processing, data exploration and train/test/split steps of our pipeline. We are working on data wrangling for the bird detection and species classification models. We have created a basic bird detector model, but we still need to optimize the model by using data wrangling techniques and tuning hyperparameters. 

We have created a demonstration notebook containing steps for data preprocessing, data exploration, train/test/split, and training the bird detector. 

Open the [Colab link](https://colab.research.google.com/drive/1wU5k5jI9TlPWy3CzXb4gabZ__YB-Cp97?usp=sharing) to run the demonstration notebook.
