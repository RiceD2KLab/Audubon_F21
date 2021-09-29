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
    <li><a href="#installation"> ➤ Installation Instructions</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#roadmap"> ➤ Roadmap</a></li>
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
* Detectron
* Keras

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
    ├── dataloader.py
    ├── data_exploration.py  
    ├── cropping.py
    ├── train_net.py

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="installation"> Installation Instructions</h2>
<p> 
  <ol>
  <li>Clone the repository</li>
  <li>Execute 'pip install requirements.txt'</li>
  <li>Execute the files wanted.</li>
  </ol>
</p> 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATASET -->
<h2 id="dataset"> Dataset</h2>
<p> 
  Houston Audubon has provided us a 52 GB image dataset consisting of images captured using DJI M300RTK UAV with a P1 camera attachment. The images are typically 8192 x 5460 high-resolution images. The dataset contains 1.5GB annotated images with corresponding CSV files for each image specifying species labels and bounding box locations. The annotated dataset features 4861 birds of 15 species, and the remaining 50.5 GB are raw images without annotations. The CSV files contain:
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

<!-- ROADMAP -->
<h2 id="roadmap"> Roadmap</h2>

<p align="justify"> 
  Weiss et. al. has trained three models namely Decision Tree, k-Nearest Neighbors, and Random Forest for human activity classification by preprocessing the raw time series data using statistical feature extraction from segmented time series. 
  The goals of this project include the following:
<ol>
  <li>
    <p align="justify"> 
      Train the same models - Decision Tree, k Nearest Neighbors, and Random Forest using the preprocessed data obtained from topological data analysis and compare the
      performance against the results obtained by Weiss et. al.
    </p>
  </li>
  <li>
    <p align="justify"> 
      Train SVM and CNN using the preprocessed data generated by Weiss et. al. and evaluate the performance against their Decision Tree, k Nearest Neighbors, and Random Forest models.
    </p>
  </li>
</ol>
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
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- EXPERIMENTS -->
<h2 id="experiments"> Experiments</h2>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- RESULTS AND DISCUSSION -->
<h2 id="results-and-discussion"> Results and Discussion</h2>



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- REFERENCES -->
<h2 id="references"> References</h2>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> Contributors</h2>

<p>
  
  <b>Krish Kabra</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="">@</a> <br>
  
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
