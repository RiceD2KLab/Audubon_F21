import glob, os
from cv2 import imread
import pandas as pd

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_bird_only_dicts(data_dir,img_ext='.JPG'):
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
    height, width = imread(file_img).shape[:2]
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


def get_bird_species_dicts(data_dir,class_names,img_ext='.JPG',unknown_bird_category=True,skip_empty_imgs=True):
  """
  Format dataset to detectron2 standard format. 
  INPUTS: 
    data_dir -- directory containing dataset files 
    img_ext -- file extension for images in dataset
    class_names -- 
    skip_empty_imgs -- keep images with no birds 
  OUTPUTS: 
    dataset_dicts -- list of dictionaries in detectron2 standard format
  """

  dataset_dicts = []

  for idx,file_csv in enumerate(glob.glob(os.path.join(data_dir,'*.csv'))):

    record = {}

    # image attributes
    root, ext = os.path.splitext(file_csv)
    file_img = root + img_ext
    height, width = imread(file_img).shape[:2]
    record["file_name"] = file_img
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width

    # annotations
    imgs_anns_df = pd.read_csv(file_csv, header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])
    # skip empty images
    if skip_empty_imgs and imgs_anns_df.shape[0]==0:
      continue
    # remove annotations for trash
    imgs_anns_df = imgs_anns_df[imgs_anns_df["class_name"]!="Trash/Debris"]

    objs = []
    for idx, row in imgs_anns_df.iterrows():
      obj = None
      for id, class_name in enumerate(class_names):
        if class_name in row["class_name"]:
          obj = {
            "bbox": [row["x"], row["y"], row["width"], row["height"]],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": id,
          }
          objs.append(obj)
          break
      if obj is None:
        if unknown_bird_category:
          obj = {
            "bbox": [row["x"], row["y"], row["width"], row["height"]],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": len(class_names)
          }
          objs.append(obj)
        else:
          continue

    record["annotations"] = objs
    dataset_dicts.append(record)

  return dataset_dicts

def register_datasets(data_dirs,img_ext,birds_species_names):
  """
  TODO: insert comment about registering datasets
  """
  for data_dir in data_dirs:
    d = os.path.basename(data_dir)
    # birds only
    if f"birds_only_{d}" in DatasetCatalog.list():
      DatasetCatalog.remove(f"birds_only_{d}")
    DatasetCatalog.register(f"birds_only_{d}", lambda d=d: get_bird_only_dicts(data_dir, img_ext))
    if f"birds_only_{d}" in MetadataCatalog.list():
      MetadataCatalog.remove(f"birds_only_{d}")
    MetadataCatalog.get(f"birds_only_{d}").set(thing_classes=["Bird"])

    # bird species
    if f"birds_species_{d}" in DatasetCatalog.list():
      DatasetCatalog.remove(f"birds_species_{d}")
    DatasetCatalog.register(f"birds_species_{d}", lambda d=d: get_bird_species_dicts(data_dir,
                                                                                  birds_species_names,
                                                                                  img_ext,
                                                                                  unknown_bird_category=True))
    if f"birds_species_{d}" in MetadataCatalog.list():
      MetadataCatalog.remove(f"birds_species_{d}")
    MetadataCatalog.get(f"birds_species_{d}").set(thing_classes=birds_species_names + ["Unknown Bird"])

    # each bird species
    for species in birds_species_names:
      if f"{species}_{d}" in DatasetCatalog.list():
        DatasetCatalog.remove(f"{species}_{d}")
      DatasetCatalog.register(f"{species}_{d}", lambda d=d: get_bird_species_dicts(data_dir,
                                                                                       [species],
                                                                                       img_ext,
                                                                                       unknown_bird_category=False))
      if f"{species}_{d}" in MetadataCatalog.list():
        MetadataCatalog.remove(f"{species}_{d}")
      MetadataCatalog.get(f"{species}_{d}").set(thing_classes=[species])
