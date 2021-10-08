import glob, os
import numpy as np 
import cv2
import pandas as pd

from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils

def get_bird_dicts(data_dir,img_ext='.JPG'): 
  """
  Format dataset to detectron2 standard format. 
  INPUTS: 
    data_dir -- directory containing dataset files 
    img_ext -- file extension for images in dataset
  OUTPUTS: 
    dataset_dicts -- list of dictionaries in detectron2 standard format
  """ 
  
  dataset_dicts = []

  for idx,file_csv in enumerate(glob.glob(os.path.join(data_dir,'*.csv'))): 
    
    record = {}

    # image attributes 
    root, ext = os.path.splitext(file_csv)    
    file_img = root + img_ext
    height, width = cv2.imread(file_img).shape[:2]
    record["file_name"] = file_img
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width
    
    # annotations 
    imgs_anns_df = pd.read_csv(file_csv, header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])
    # skip empty images
    if imgs_anns_df.shape[0]==0: 
      continue
    # remove annotations for trash 
    imgs_anns_df = imgs_anns_df[imgs_anns_df["class_name"]!="Trash/Debris"]
    
    objs = []
    for idx, row in imgs_anns_df.iterrows():  
      obj = {
          "bbox": [row["x"], row["y"], row["width"], row["height"]],      
          "bbox_mode": BoxMode.XYWH_ABS,
          "category_id": 0,
      }
      objs.append(obj)

    record["annotations"] = objs
    dataset_dicts.append(record)

  return dataset_dicts
