<h1 align="center"> Team Audubon </h1>
<h3 align="center"> Development of Machine Learning Algorithms for Precision Waterbird Monitoring </h3>  

</br>


<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Team Audubon in SP22"> ➤ Team Audubon in SP22</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#installation"> ➤ Installation & Usage Instructions</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li>
      <a href="#Data Science Pipeline"> ➤ Data Science Pipeline</a>
      <ul>
        <li><a href="#data-augmentation">Data Augmentation</a></li>
        <li><a href="#Modeling">Modeling</a></li>
        <li><a href="#Experiments">Experiments</a></li>
      </ul>
    </li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="Team Audubon in SP22"> Team Audubon in SP22</h2>

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
    ├────── (colab files)
    ├── FasterRCNNDenseNet
    ├────── (train Faster RCNN with DenseNet backbone)
    ├── Flex_Faster_RCNN
    ├────── (train Faster RCNN with flexible configurations and backbones)   
    ├── configs
    ├────── (configurations of Detectron2)
    ├── data_aug
    ├────── (data augmentation methods)
    ├── utils
    ├────── (useful functions for constructing Faster RCNN)
    ├── README.md
    ├── Audubon_S22.py
    ├── RunApp.py
    ├── Training_only.py
    ├── requirements.txt
    ├── startup.sh
    ├── train_net.py
    ├── detect_only.py
    

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="installation"> Installation & Usage Instructions</h2>

<p> 
  <ol>
  <li><b>Clone the repository</b></li>

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
  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RiceD2KLab/Audubon_F21/blob/SP22/Sp22_Audubon_Bird_Detection_Tutorial.ipynb) <br> 

See [train_net.py](train_net.py), [wandb_train_net.py](wandb_train_net.py), or [Colab Notebook](Sp22_Audubon_Bird_Detection_Tutorial.ipynb) for usage of code. 


  </ol>
</p> 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<!-- DATASET -->
<h2 id="dataset"> Dataset</h2>

<p align="center">
  <img src="https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/utils/pipeLine/Data.png?raw=true" width="600">
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

<!-- DATA SCIENCE PIPELINE -->
<h2 id="Data Science Pipeline"> Data Science Pipeline </h2>

<p align="center">
  <img src="https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/utils/pipeLine/DataPipeLine.png?raw=true" width="600">
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- STATISTICAL FEATURE -->
<h2 id="data-augmentation"> Data Augmentation</h2>

<p align="justify"> 

  Deep learning models are effective with about 1,000 images per class, but some bird species do not have abundant training samples in our dataset. Our team plans to make deep learning models more robust via data augmentation, which means training models with synthetically modified data:
  <ul>
  <li><b>data cropping: </b> namely extracting the target block of each image.</li>
  <li><b>rotation: </b>Orthogonal or non-orthogonal rotations. Rotation is a natural data augmentation step for our data at hand because the bird images are taken from different angles by drones.</li>
  <li><b>flipping: </b> Houston Audubon bird images are taken from different angles by drones, so it is natural to adopt horizontal or vertical image flipping for detection convenience.</li>
  <li><b>color space: </b> Randomly tuning brightness and contrast of the bird images can compensate detection bias caused by different light, weather and location conditions.</li>
    
  </ul>

These data augmentation steps help models adapt to different orientations, locations, light conditions and scales of the
same object class, and will boost the performance of the models.

<b> For the time being, our model is only trained on original data. </b> We plan to retrain our model on the augmented dataset and compare performances. We are generating a larger training set using the augmentation methods mentioned above. Specifically, both the original images and the transformed images will be fed to the model in the training phase,
but only original images will be used for evaluation and testing purposes.

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- STATISTICAL FEATURE -->
<h2 id="Modeling"> Modeling</h2>

<li><b>Faster R-CNN</b></li>

<p align="center">
  <img src="https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/Flex_Faster_RCNN/fasterRCNN.png">
</p>

 <li><b>Bayesian Hyperparameters tuning</b></li>
Model performance is heavily influenced by hyperparameters. Grid search or manual tuning both necessitate a lot of tuning skill and computational resources, which is why we employ bayesian hyperparameters tuning to address these two issues.This model’s entire procedure is defined as follows:
  <ul>
  <li><b>(1)</b> To describe the uncertainty, compute the posterior belief u(x) using a surrogate Gaussian pro- cess to create an estimate of the mean and standard deviation around this estimate σ(x).</li>
  <li><b>(2)</b>Calculate an acquisition function a(x) that is proportional to how advantageous it is to sample the next point in the range of values.</li>
  <li><b>(3)</b> Locate the maximum point of this acquisition function and sample from there.</li>
  <li><b>(4)</b> This process is repeated a certain number of times, commonly known as the optimization bud- get, until a pretty good point is reached.</li>
    
  </ul>
  
 <li><b>Weighted loss function</b></li>
Given that our dataset is unbalanced and has to be corrected. The idea is to give distinct classes in the loss function sample size-based weights so that the model can focus more on the minority classes during training. We propose a custom weighted Cross Entropy loss function for the network's classification layer:

<p align="center">
  <img src="https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/utils/pipeLine/WeightedLoss.png" width="300">
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- EXPERIMENTS -->
<h2 id="Experiments"> Experiments</h2>

<p align="justify"> 

We utilize Faster R-CNN module with a ResNet-50-FPN backbone. 
We first split dataset into train set(70%) validation set(20%) and test set(10%).
We then use train set and validation set for bayesian optimization to get the optimal hyperparameters.
Next, we train our model with new hyperparameters and evaluate it on test set.

<i><b>Note:</b> The model weights used to initialize both the bird-only and bird-species detector come from a pre-trained model on the MS COCO dataset. </i>

<ol>
  <li><b>Object Detection Results</b></li> 
  <p align="center">
    <img src="https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/utils/pipeLine/Results.png" width="700">
  </p>

  The high precision scores for all bird species using an IoU threshold of 0.50 is excellent, except for the minority category (“Roseate Spoonbill” and "White Ibis"), where the model drastically fails to classify. 
  
  <li><b>Experiment Example</b></li> 
    <p align="center">
    <img src="https://github.com/RiceD2KLab/Audubon_F21/blob/SP22/utils/pipeLine/Cover%2520Image.JPEG">
  </p>
</ol>

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> Contributors</h2>

<p>
  
  <b>Wenbin Li</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: wl56@rice.edu <a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/lwb56">@Wenbin</a> <br>
  
  <b>Tianjiao Yu</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: ty37@rice.edu<a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/TianjiaoYu">@Tianjiao</a> <br>

  <b>Raul Garcia</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: rg66@rice.edu<a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/raulgarcia66">@Raul</a> <br>

  <b>Dhananjay Vijay Singh</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: Dv17@rice.edu<a></a> <br>

  <b>Jiahui Yu</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: jy89@rice.edu<a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/KarenYu729">@Jiahui</a> <br>

  <b>Maojie Tang</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: mt84@rice.edu<a></a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/WaitDumplings">@Maojie</a> <br>
  
  <b>Hank Arnold</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: hmarnold@msn.com<a></a> <br>
</p>
<br>
✤ <i>This was the project for the course COMP 449/549 - Machine Learning and Data Science Projects (Spring 2022), at <a href="https://www.rice.edu/">Rice University</a><i>
