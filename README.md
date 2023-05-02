# AviAlert - D2K Team Audubon Spring 2023

The name "AviAlert" is a combination of the words "aviary" (a place where birds are kept) and "alert" (to bring attention to something). AviAlert is a Python package for object dectection and classification.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Demo

Run the demo notebook in Colab by clicking [here](https://colab.research.google.com/drive/1TiGTjLM1XzjMOdD-v-PrEPylSTFlBXjL?usp=sharing). The executed notebook is saved in `notebooks/demo.ipynb`.

## Project description

Audubon Texas is pioneering the use of drone photography to monitor populations of colonial waterbirds in the Texas Gulf Coast. Researchers need to comb through high resolution images and manually count up birds by species class, which can take weeks for even one image. Seeking to automate the annotation process, Houston Audubon partnered with the Data to Knowledge (D2K) lab at Rice University. Student teams have developed an object detection based deep learning model that can automatically detect birds within a drone image and classify their species. This semester, we will continue the project with two main objectives:

    1. Improve the species classification capabilities of the model.
    2. Develop an AI-assisted waterbird annotation tool.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Prerequisites
The following open source packages are used in this project:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `torch`
  - `torchvision`
  - `livelossplot`
  - `split-folders`
  - `cv2`
  - `pycocotools`

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Folder structure

  - `assets` || Directory containing files used in the README, such as our data science pipeline
  - `GUI` || Directory containing files used to run the GUI
    - `audubon-website` || Contain files for the front-end of the GUI
        - `README.md` || Readme file for the GUI. Refer here for details of the GUI
        - `package.json`
        - `src`|| Directory containing files for the front-end of the GUI
    - `server` || Directory containing files for setting up the server and GUI backend
  - `notebooks` || Directory containing Jupyter notebooks for demonstrations or examples
    - `demo.ipynb`
  - `src` || Directory containing helper functions for data processing, data exploration and modeling
    - `data` || Contains helper functions for data processing and data exploration
       - `coco` || Contains helper functions for COCO API. Sources: Microsoft and Torchvision. For more information, see the bottom of the README
         - `coco_eval.py`
         - `coco_utils.py`
         - `interface.py`
         - `mask.py`
         - `transforms.py`
         - `utils.py`
      - `convert_annotations.py`
      - `crop_birds.py`
      - `dataloader.py`
      - `plotlib.py`
      - `transforms.py`
      - `utils.py`
    - `loss_fun` || Contains helper functions for implemnting a weighted cross-entropy loss function during model training
      - `weighted_cross_entropy.py`
    - `models` || Contains helper functions for using a pre-trained ResNet50 model for the classifier
      - `pretrained.py`
    - `optimizers` || Contains helper functions for implementing optimizers for model training
      - `adam.py`
      - `sgd.py`
    - `eval.py`
    - `train.py`
  - `config.py`|| File containing constant parameters, such as column names, bird classes and hyperparameters
  - `requirements.txt` || List of dependencies 
  - `train_classifier.py`|| Function for training a bird classification model
  - `train_detector.py` || Function for training a bird localization model
  - `update_database.py`|| Function for setting up data paths
  - `README.md`

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Installation instructions

 ### Clone the repository

  ```linux
  !git clone -b avialert https://github.com/RiceD2KLab/Audubon_F21.git
  ```
 ### Install dependencies

  The Python version is 3.8.16 and users can install the required packages using the following command:

  ```linux
  pip install -r requirements.txt
  ```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Data science pipeline
Our data science pipeline splits the tasks of bird detection and species classification between two models. The bird detection model is based on the model developed by previous groups. We focused on developing a bird classification model that performs well for all species classes. For each model, we followed specific data wrangling, model training and evaluation steps. Additionally, we overhauled the output of the models by launching a graphical user interface (GUI) which will allow scientists to easily annotate their own drone images

<img src="assets/data science pipeline.png">

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

In order to access the full dataset in Colab, use the following code:

```
!gdown -q 'https://drive.google.com/uc?id=1oK_1Y16dMwcqytFd41UNHxP_O7ju22bo'
!unzip -q './S23-Audubon.zip' && mv S23-Audubon [desired folder in working directory] 
```

The full Audubon dataset is stored in Google Drive, and must be accessed using the !gdown command. 

Our team utilize Dropbox which permits multiple downloads of files within a short timeframe. We have segregated the images and annotations into distinct folders for the original and tiled data. Use the following code to retrieve the data:

```linux
!wget 'https://www.dropbox.com/s/xi6ipd06nqyq3k5/database.zip'
!unzip -q './database.zip'
```

## Data format

CSV and JPG files that share the same name but have different file extensions. The CSV files and JPG files should be saved in different folders.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Reference

Files in `scr\data\coco` are downloaded from [COCO API](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) and [PyTorch reference for detection](https://github.com/pytorch/vision/tree/main/references/detection).
