<h1 align="center"> Houston Audubon </h1>
<h3 align="center"> Development of Machine Learning Algorithms for Precision Waterbird Monitoring </h3>  

</br>


<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ About The Project</a></li>
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
<h2 id="about-the-project"> About The Project</h2>

<p align="justify"> 
  In order to both improve the accuracy of bird counts as well as the speed, Houston Audubon has asked us to develop machine learning and computer vision algorithms for the detection of birds using images from UAVs, with the specific goals to:
  <ol> 
  <li> Count and survey the number of birds.
  <li> Identify different species of detected birds.
  <li> Determine if detected birds are adults or chicks.
  <li> Count the number of nests in the UAV images.
</ol>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>

<!--This project is written in Python programming language. <br>-->
The following open source packages are used in this project:
* Numpy
* Pandas
* Matplotlib
* OpenCV 
* Detectron2
* PyTorch
* WAndB

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<!-- │   ├── raw_data
│   │   ├── phone
│   │   │   ├── accel
│   │   │   └── gyro
│   │   ├── watch
│   │       ├── accel
│   │       └── gyro
│   │
│   ├── transformed_data
│   │   ├── phone
│   │   │   ├── accel
│   │   │   └── gyro
│   │   ├── watch
│   │       ├── accel
│   │       └── gyro
│   │
│   ├── feature_label_tables
│   │    ├── feature_phone_accel
│   │    ├── feature_phone_gyro
│   │    ├── feature_watch_accel
│   │    ├── feature_watch_gyro
│   │
│   ├── wisdm-dataset
│        ├── raw
│        │   ├── phone
│        │   ├── accel
│        │   └── gyro
│        ├── watch
│            ├── accel
│            └── gyro -->
<h2 id="folder-structure"> Folder Structure</h2>

    code
    .
    │
    ├── data
    │
    ├── README.md
    ├── requirements.txt
    ├── data_exploration.py  
    ├── cropping.py
    ├── train_net.py

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="installation"> Installation & Usage Instructions</h2>
<p> 
  <ol>
  <li>Clone the repository</li>

  ```linux
  git clone https://github.com/RiceD2KLab/Audubon_F21.git
  ```

  <li>Install Detectron2</li>
  Requirements: Linux or macOS with Python ≥ 3.6

  ```linux
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  # (add --user if you don't have permission)

  # Or, to install it from a local clone:
  git clone https://github.com/facebookresearch/detectron2.git
  python -m pip install -e detectron2

  # On macOS, you may need to prepend the above commands with a few environment variables:
  CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
  ```

  <li>Install Pytorch</li>
  Requirements: Linux or macOS with Python ≥ 3.6

  ```linux
  pip3 install torch torchvision
  ```

  <li>Install other dependencies</li>

  ```linux
  pip install requirements.txt
  ```

  <li>Execute the scripts as required.</li>
  </ol>
</p> 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATASET -->
<h2 id="dataset"> Dataset</h2>
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

For the time being, our model is only trained on original data. We plan to retrain our model on the augmented dataset and compare performances. We are generating a larger training set using the augmentation methods mentioned above. Specifically, both the original images and the transformed images will be fed to the model in the training phase, but only original images will be used for evaluation and testing purposes.

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- EXPERIMENTS -->
<h2 id="experiments"> Experiments</h2>

<p align="justify"> 

For our initial testing, we choose to train a single RetinaNet module with a ResNet-50-FPN backbone. We first train our model to perform the simple task of detecting birds with no distinction of species. 

We then train the model to identify bird species: namely, Brown Pelicans, Laughing Gulls, and Mixed Terns. Due to the lack of annotated data available for other bird species, we re-label all other bird species under the "Other/Unknown" category. 

<i><b>Note:</b> The model weights used to initialize the bird-only detector come from a pre-trained RetinaNet on the COCO dataset, and the weights used to initialize the bird species detector come from the pre-trained bird-only detector. </i>

<ol>
  <li><b>Bird-only detector</b></li> 
      
  |                 | Birds |
  |-----------------|-------|
  | AP (IoU = 0.5)  | 93.7% |
  | AP (IoU = 0.75) | 26.4% |
  | mAP             | 43.7% |

  The high AP of 93.7% using an IoU threshold of 0.50 is very promising.
  
  The mAP of 43.7% is comparableto the state-of-the-art results for challenging object detection tasks such as on the COCO dataset.
  <li><b>Bird species detector</b></li>

  |                | Brown Pelican | Laughing Gull | Mixed Tern | Other/Unknown | Overall |
  |----------------|---------------|---------------|------------|---------------|---------|
  | AP (IoU = 0.5) | 87.1%         | 91.6%         | 81.6%      | 30.1%         | 72.6%   |
  | mAP            | 43.6%         | 33.1%         | 40.7%      | 11.9%         | 32.3%   |

  The lower AP of 72.6% using an IoU threshold of 0.50 in comparison to the bird-only detector is problematic. In particular, the model drastically fails to classify “Other/Unknown” birds. We see that the model achieves mediocre performance in detecting birds, with several birds remaining undetected or being classified incorrectly. Although the model  weights were initialized to those of  the trained bird-only detector, the network has unlearnt its ability to localize birds adequately.
</ol>

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- RESULTS AND DISCUSSION -->
<h2 id="results-and-discussion"> Results and Discussion</h2>

<p>
Leveraging RetinaNet, we trained a bird-only detection model as well as a bird species detection model that classified Brown Pelicans, Laughing Gulls, and Mixed Terns. As observed in the results, for the bird-only detector we achieved a mean average precision of 43.7% across the 11 different IoU thresholds, which is comparable to state-of-the-art results achieved by object detectors on benchmark computer vision datasets. Across the bird species detector model, we observe a mean average precision of 32.3%, which is significantly worse. We have some next steps that we hope will help improve the mAP precision results.
</p>



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- REFERENCES -->
<h2 id="references"> References</h2>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> Contributors</h2>

<p>
  
  <b>Krish Kabra</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: krish.kabra@rice.edu <a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/krishk97">@krishk97</a> <br>
  
  <b>Minxuan Luo</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="">@</a> <br>

  <b>William Lu</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="">@</a> <br>

  <b>Alexander Xiong</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="">@</a> <br>
</p>

<br>
✤ <i>This was the project for the course COMP 449 - Machine Learning and Data Science Projects (Fall 2021), at <a href="https://www.rice.edu/">Rice University</a><i>
