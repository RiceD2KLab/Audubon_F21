<h1 align="center"> Team Audubon </h1>
<h3 align="center"> Development of Machine Learning Algorithms for Precision Waterbird Monitoring </h3>  

</br>


<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Team Audubon in 22SP"> ➤ Team Audubon in 22SP</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#installation"> ➤ Installation & Usage Instructions</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li>
      <a href="#preprocessing"> ➤ Preprocessing</a>
      <ul>
        <li><a href="#tiling">Tiling</a></li>
        <li><a href="#data-augmentation">Data Augmentation</a></li>
      </ul>
    </li>
    <!--<li><a href="#experiments">Experiments</a></li>-->
    <li><a href="#results-and-discussion"> ➤ Results and Discussion</a></li>
    <li><a href="#references"> ➤ References</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="Team Audubon in 22SP"> Team Audubon in 22SP</h2>

<p align="justify"> 
  In order to both improve the accuracy of bird counts as well as the speed, Houston Audubon and students from the D2K capstone course at Rice University
  develop machine learning and computer vision algorithms for the detection of birds using images from UAVs, with the specific goals to:
  <ol> 
  <li> Count and survey the number of birds.
  <li> Identify different species of detected birds.
</ol>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
<img src="https://raw.githubusercontent.com/numpy/numpy/9ee47e0ebe7e869f4ddcf1e3d18978fa23d43c1d/branding/logo/primary/numpylogo.svg" width="140">
<img src="https://pandas.pydata.org/static/img/pandas.svg" width="150">
<img src="https://matplotlib.org/_static/logo2.svg" width="150">
<img src="https://github.com/opencv/opencv/blob/f86c8656a3bfa9219359faba16fd11091fbb7938/doc/js_tutorials/js_assets/opencv_logo.jpg?raw=true" width="75">
<img src="https://raw.githubusercontent.com/facebookresearch/detectron2/main/.github/Detectron2-Logo-Horz.svg" width="200">

<!--This project is written in Python programming language. <br>-->
The following open source packages are used in this project:
* Numpy
* Pandas
* Matplotlib
* OpenCV 
* Detectron2

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="folder-structure"> Folder Structure</h2>

    code
    .
    ├── .ipynb_checkpoints
    ├────── (..........)
    ├── FasterRCNNDenseNet
    ├────── (Train Faster RCNN with DenseNet model)
    ├── Flex_Faster_RCNN
    ├────── (Train Faster RCNN with flexible configurations and backbones)   
    ├── configs
    ├────── (configurations of Detectron2)
    ├── data_aug
    ├────── (data augmentation methods)
    ├── utils
    ├────── (useful functions for constructing Faster RCNN)
    ├── README.md
    ├── requirements.txt
    ├── data_exploration.py  
    ├── Audubon-Bird-Detection-Tutorial.ipynb
    ├── train_net.py

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="installation"> Installation & Usage Instructions</h2>

<p> 
  <ol>
  <li>Clone the repository</li>

  ```linux
  git clone https://github.com/RiceD2KLab/Audubon_F21.git
  ```
  <li><b>Install Pytorch</b></li>
<a href="https://pytorch.org/get-started/locally/">Installation instructions here</a> <br>
  Requirements: Linux or macOS with Python ≥ 3.6

  ```linux
  pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
  ```

  <li> <b> Install Detectron2 </b> </li>
  <a href="https://detectron2.readthedocs.io/en/latest/tutorials/install.html">Installation instructions here</a> <br>
  Requirements: Linux or macOS with Python ≥ 3.6 <br>
  For Windows: Detectron2 is continuously built on Windows with CircleCI. However, official support for it is not provided.

  ```linux
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  # (add --user if you don't have permission)

  # Or, to install it from a local clone:
  git clone https://github.com/facebookresearch/detectron2.git
  python -m pip install -e detectron2

  # On macOS, you may need to prepend the above commands with a few environment variables:
  CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
  ```


  <li>Install other dependencies</li>

  ```linux
  pip install requirements.txt
  ```

  <li>Execute the scripts as required.</li>
  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/Sp22_Audubon_Bird_Detection_Tutorial.ipynb) <br> 

See [train_net.py](train_net.py), [wandb_train_net.py](wandb_train_net.py), or [Colab Notebook](Sp22_Audubon_Bird_Detection_Tutorial.ipynb) for usage of code. 


  </ol>
</p> 

![-----------------------------------------------------](https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/utils/pipeLine/DataPipeLine.png?raw=true)

<!-- DATA SCIENCE PIPELINE -->
<h2 id="dataset"> Data Science Pipeline </h2>

<p align="center">
  <img src="assets/pipeline.png" width="600">
</p>

![-----------------------------------------------------](https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/utils/pipeLine/Data.png?raw=true)

<!-- DATASET -->
<h2 id="dataset"> Dataset</h2>

<p align="center">
  <img src="assets/DSC06695 - Ref Image.jpg" width="600">
</p>

<p> 
  Houston Audubon has provided us a 52 GB image dataset consisting of images captured using DJI M300RTK UAV with a P1 camera attachment. The images are typically 8192 x 5460 high-resolution images. The dataset contains 3 GB annotated images with corresponding CSV files for each image specifying species labels and bounding box locations. The annotated dataset features 19276 birds of 15 species, and the remaining 50.5 GB are raw images without annotations. The CSV files contain:
  <ul>
    <li><b>species id</b>: unique species id in integer</li> 
    <li><b>species label</b>: species label in words</li> 
    <li><b>x</b>: x min of a bounding box</li> 
    <li><b>y</b>: y min of a bounding box</li> 
    <li><b>width</b>: width of a bounding box</li> 
    <li><b>height</b>: height of a bounding box</li> 
  </ul>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREPROCESSING -->
<h2 id="preprocessing"> Preprocessing</h2>

<p align="justify"> 
  The data wrangling module of the pipeline largely involves preparing the data to be fed into deep learning models used to detect objects, namely birds. Our data wrangling process includes:
  <ol>
    <li><b>Tiling</b></li> 
    <li><b>Data Augmentation</b></li>
  </ol>

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PRE-PROCESSED DATA -->
<h2 id="tiling"> Tiling</h2>

<p align="justify"> 
  Principally, deep learning models train faster and have better performances on smaller images. For instance, 600 × 600 pixels is usually an ideal image size for typical object detection deep learning models. Therefore, our first attempt was to split the 8192 × 5460 images into tiles. The size of generated images can be specified by setting parameters and is default to be 600 × 600.
  
  A caveat of this approach is that unavoidably some birds will be cut into two parts and appear in two neighboring patches, as seen in Figure 2. In addition, as counting the number of birds is among the objectives, the same problem needs to be tackled in the detection phase as well. In this case, only the generated image with over 50% fraction of the cropped bird keeps the bounding box, while the remaining fraction of the bounding box in another image is discarded. This means that we are training the model to detect both complete birds and partial birds.
  
  In the detection stage, we will also try to come up with a proper merging mechanism to merge partial detection in neighboring patches and count as one if repeated counting is a common pattern in detection.
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- STATISTICAL FEATURE -->
<h2 id="data-augmentation"> Data Augmentation</h2>

<p align="justify"> 

  Deep learning models are effective with about 1,000 images per class, but some bird species do not have abundant training samples in our dataset. Our team plans to make deep learning models more robust via data augmentation, which means training models with synthetically modified data:
  <ul>
  <li><b>rotation: </b>Orthogonal or non-orthogonal rotations. Rotation is a natural data augmentation step for our data at hand because the bird images are taken from different angles by drones.</li>
  <li><b>random crop: </b>Randomly sample a section from the image and resize it to the original image size.</li>
  </ul>

These data augmentation steps help models adapt to different orientations, locations, and scales of the
same object class, and will boost the performance of the models.

We utilized the <i>imgaug</i> library to generate modified images. We have tried several types of augmentations: flipping, blurring, adding Gaussian noise and changing color contrasts. 

<b> For the time being, our model is only trained on original data. </b> We plan to retrain our model on the augmented dataset and compare performances. We are generating a larger training set using the augmentation methods mentioned above. Specifically, both the original images and the transformed images will be fed to the model in the training phase,
but only original images will be used for evaluation and testing purposes.

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- EXPERIMENTS -->
<h2 id="experiments"> Experiments</h2>

<p align="justify"> 

We utilize a RetinaNet and Faster R-CNN module both with a ResNet-50-FPN backbone. 
We first train our model to perform the simple task of detecting birds with no distinction of species.
We then train the model to identify bird species: namely, Brown Pelicans, Laughing Gulls, Mixed Terns, Great Blue Herons, and Great Egrets/White Morphs. 

Due to the lack of annotated data available for other bird species, we re-label all other bird species under the "Other/Unknown" category. 

<i><b>Note:</b> The model weights used to initialize both the bird-only and bird-species detector come from a pre-trained model on the MS COCO dataset. </i>

<ol>
  <li><b>Bird-only detector (RetinaNet ResNet-50 FPN)</b></li> 
      
  |                 | Birds |
  |-----------------|-------|
  | AP (IoU = 0.5)  | 93.7% |
  | AP (IoU = 0.75) | 26.4% |
  | mAP             | 43.7% |

  The high AP of 93.7% using an IoU threshold of 0.50 is very promising.
  
  The mAP of 43.7% is comparableto the state-of-the-art results for challenging object detection tasks such as on the COCO dataset.
  <li><b>Bird species detector (Faster R-CNN ResNet-50 FPN)</b></li>

  |                | Brown Pelican | Laughing Gull | Mixed Tern | Great Blue Heron | Great Egret/White Morph | Other/Unknown | Overall |
  |----------------|---------------|---------------|------------|------------------|-------------------------|--------------|---------|
  | AP (IoU = 0.5) | 98.8%         | 100.0%        | 97.6%      | 98.5%            | 96.9%                   | 0.0%         | 82.0%   |

  The higher AP for all bird species using an IoU threshold of 0.50 in comparison to the bird-only detector is excellent, except for the “Other/Unknown” categroy, where the model drastically fails to classify. 
  Nevertheless, we can combine the results from a bird-only detector and bird-species detector to recover the poor performance of the "Other/Unknown" bird category.
</ol>

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> Contributors</h2>

<p>
  
  <b>Krish Kabra</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: krish.kabra@rice.edu <a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/krishk97">@krishk97</a> <br>
  
  <b>Minxuan Luo</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: ml122@rice.edu<a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/minxuanluo">@minxuanluo</a> <br>

  <b>Alexander Xiong</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: xionga27@rice.edu<a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/awx1">@awx1</a> <br>

  <b>William Lu</b> wyl1@rice.edu<br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="">@</a> <br>

  <b>Anna Vallery</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: avallery@houstonaudubon.org<a></a> <br>

  <b>Richard Gibbons Lu</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: rgibbons@houstonaudubon.org<a></a> <br>
  
  <b>Hank Arnold</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: hmarnold@msn.com<a></a> <br>
</p>
<br>
✤ <i>This was the project for the course COMP 449/549 - Machine Learning and Data Science Projects (Fall 2021), at <a href="https://www.rice.edu/">Rice University</a><i>
