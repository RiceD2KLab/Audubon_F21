<h1 align="center"> Team Audubon @Rice</h1>
<h3 align="center"> Development of Machine Learning Algorithms for Precision Waterbird Monitoring </h3>  

</br>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Team Audubon in FL22"> ➤ Team Audubon in FL22</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#installation"> ➤ Usage Instructions</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#Data Science Pipeline"> ➤ Data Science Pipeline</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="Team Audubon in FL22"> Team Audubon in FL22</h2>

<p align="justify"> 
  In order to improve the accuracy and speed of bird counts, Texas Audubon and students from the D2K capstone course at Rice University
  develop machine learning and computer vision algorithms for the detection of birds using images from UAVs, with the specific goals to:
  <ol> 
  <li> Extend previous model to identify visually similar birds.
  <li> Try to interpret the model, especially on failed cases.
</ol>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> Prerequisites</h2>

<p align="justify"> 
  Our implementation is based on Python 3.7, Detectron2, PyTorch, and a list of packages that can be found in <a href="./requirements.txt">./requirements.txt</a>. This file may be used to create an environment using:
</p>
    
  ```linux
  $ conda create --name <env> --file ./requirements.txt
  ```
<p align="justify"> 
  We tested this project on a Ubuntu 20.04.3 LTS machine with a GeForce RTX 2080 Ti GPU.
</p>    

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2 id="folder-structure"> Folder Structure</h2>

    this repo
    .
    ├── data/
    ├────── (dataset of the original UAV and the processed images) 
    ├── output/
    ├────── (location to save the trained models)   
    ├── utils/
        ├── (useful functions for constructing Faster RCNN)
        ├── config.py          #for Detecron2 configuration
        ├── cropping.py        #for data processing and preparation
        ├── augmentation.py    #for data augmentation
        ├── dataloader.py      #for Detecron2 data manipulation
        ├── hyperparameter.py  #for hyperparameter tuning and training
        ├── custom_loss.py     #for customized loss functions, e.g., weighted loss
        ├── trainer.py         #for Detecron2 training handling
        ├── evaluation.py      #for result evaluation
        ├── confmat.py         #for confusion matrix computation
        ├── gradcam.py         #for interpretable visualization
        ├── pipeline2.py       #for detector-classifier pipeline that corrects detected tern labels
    ├── README.md
    ├── requirements.txt
    ├── main_ms_detector.py    # main function entry for training the detector
    ├── MultiSpec-MonitorEvalCAM.ipynb    # visualize training results and evaluation
    ├── Terns-FullPipelineWithClas.ipynb  # detector-classifier pipeline implementation
  

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
    
<p> 
    For a more detailed view of the expected dataset organization, please refer to <a href="./main_ms_detector.py">./main_ms_detector.py</a>.
    To get access to our full dataset, please contact <a href="https://tx.audubon.org">Audubon Texas</a>.
    Our trained models are available on <a href="https://drive.google.com/drive/folders/1F_AHuyQ9VVhkrR15tnk4KWnBJoQrvlcw?usp=share_link">Google Drive</a>.
</p>


<h2 id="installation"> Installation & Usage Instructions</h2>

<p> 
  <ol>
  <li><b>Clone the repository (the latest branch) and direct to the local folder.</b></li>

  ```linux
  $ git clone -b FL22 https://github.com/RiceD2KLab/Audubon_F21.git
  $ cd Audubon_F21
  ```
      
  <li><b>Execute the main script to train a detector.</b></li>
      
  ```linux
  $ python main_ms_detector.py
  ```
      
  <li><b>View and evaluate training results.</b></li>
      
  ```linux
  $ jupyter botebook MultiSpec-MonitorEvalCAM.ipynb
  ```
      
  <li><b>Try out the 2-stage pipeline for terns.</b></li>
      
  ```linux
  $ jupyter botebook Terns-FullPipelineWithClas.ipynb
  ```
      
  </ol>
</p> 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATASET -->
<h2 id="dataset"> Dataset</h2>

<p> 
  In Fall 21 and Spring 22, Houston Audubon provided a 50+ GB image dataset consisting of images captured using DJI M300RTK UAV with a P1 camera attachment. The images are typically 8192 x 5460 high-resolution images. The dataset contains 3 GB human-annotated images with corresponding CSV files for each image specifying species labels and bounding box locations. The annotated dataset features 19276 birds of 15 species. The CSV files contain:
  <ul>
    <li><b>species id</b>: unique species id in integer</li> 
    <li><b>species label</b>: species label in words</li> 
    <li><b>x</b>: x min of a bounding box</li> 
    <li><b>y</b>: y min of a bounding box</li> 
    <li><b>width</b>: width of a bounding box</li> 
    <li><b>height</b>: height of a bounding box</li> 
  </ul>

In Fall 22, Houston Audubon provided us with a smaller 300 MB dataset containing 10 images generated using a photogrammetry process. 
Each image is 10k x 10k high resolution with corresponding annotation files that feature 4 bird classes: Royal Terns, Sandwich Terns, non-nesting Royal Terns, and non-nesting Sandwich Terns. 
The annotation files consist of CSV files in the same format detailed above for bounding boxes as well as new CSV files containing indicator points for each labeled bird. 
The purpose of this dataset is to assist the training process with the fine-grained classification problem of Mixed Terns.
  
In this project, we use data from both old and new datasets.
The aggregated dataset features more than 30 unique classes since many species were further divided into different classes based on the year and/or shape of the bird, e.g., adult vs chick, wings spreading vs flying, etc. 
Some classes contain only few instances. 
We cleaned up the data and ended up with over 20 classes (including distinct Royal and Sandwich Terns) on which we trained a multi-species bird detector. 
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATA SCIENCE PIPELINE -->
<h2 id="Data Science Pipeline"> Data Science Pipeline </h2>

<p align="center">
  <img src="https://github.com/RiceD2KLab/Audubon_F21/blob/FL22/utils/pipeLine/Pcovimg.JPEG?raw=true" width="400">
</p>

<p> For more details, please refer to our report. *Coming soon!* </p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> Contributors</h2>

<p>
  <b>Students:</b> Tony Gao (tg27@rice.edu), Christopher Le (ctl7@rice.edu), Boning Li (bl41@rice.edu), Linfeng Lou (ll90@rice.edu), Haixiao Wang (hw68@rice.edu)<br>
  <b>Mentor:</b> Krish Kabra (krish.kabra@rice.edu)<br>
  <b>Sponsors:</b> Hank Arnold (hmarnold@msn.com), Richard Gibbons (rgibbons@houstonaudubon.org), Anna Vallery (avallery@houstonaudubon.org)<br>
</p>
<br>
✤ <i>This was the project for the course COMP 449/549 - Machine Learning and Data Science Projects (Spring 2022), at <a href="https://www.rice.edu/">Rice University</a><i>
