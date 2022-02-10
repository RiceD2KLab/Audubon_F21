import glob, os
from cv2 import imread
import pandas as pd
import boto3

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_bird_only_dicts_AWS(s3client, my_bucket, data_key, img_ext='.JPG'):
    """
    Format dataset to detectron2 standard format. 
    INPUTS: 
    data_dir -- directory containing dataset files 
    img_ext -- file extension for images in dataset
    OUTPUTS: 
    dataset_dicts -- list of dictionaries in detectron2 standard format
    """
    
    dataset_dicts = []
    
    # get all the filenames for the annotated files on the S3 bucket
    files = [x['Key'] for x in s3client.list_objects_v2(Bucket=my_bucket, Prefix=data_key) ['Contents'] if x['Key'].split('.')[-1] == 'csv'] 
    
    for idx,file_csv in enumerate(files): 
        record = {}

        imre = s3client.get_object(Bucket = my_bucket, Key = file_csv.replace('csv', 'JPEG'))
        im = Image.open(imre['Body'])
        height, width, rgb = np.array(im).shape
        record["file_name"] = file_csv.replace('csv', 'JPEG')
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # annotations

        annt = s3client.get_object(Bucket = my_bucket, Key = file_csv)

        imgs_anns_df = pd.read_csv(annt['Body'], header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])
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


def get_bird_species_dicts_AWS(s3client, my_bucket, data_key,class_names,img_ext='.JPG',unknown_bird_category=True,skip_empty_imgs=True):
  """
  Format dataset to detectron2 standard format. 
  INPUTS: 
    data_dir -- directory containing dataset files 
    img_ext -- file extension for images in dataset
    class_names -- names of bird species
    skip_empty_imgs -- keep images with no birds 
  OUTPUTS: 
    dataset_dicts -- list of dictionaries in detectron2 standard format
  """

    dataset_dicts = []
    
    # get all the filenames for the annotated files on the S3 bucket
    files = [x['Key'] for x in s3client.list_objects_v2(Bucket=my_bucket, Prefix=data_key) ['Contents'] if x['Key'].split('.')[-1] == 'csv']
    
    for idx,file_csv in enumerate(files): 
        record = {}

        # image attributes
        imre = s3client.get_object(Bucket = my_bucket, Key = file_csv.replace('csv', 'JPEG'))
        im = Image.open(imre['Body'])
        height, width, rgb = np.array(im).shape
        record["file_name"] = file_csv.replace('csv', 'JPEG')
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # annotations

        annt = s3client.get_object(Bucket = my_bucket, Key = file_csv)

        imgs_anns_df = pd.read_csv(annt['Body'], header=0, names = ["class_id", "class_name", "x", "y", "width", "height"])

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
                    obj = {"bbox": [row["x"], row["y"], row["width"], row["height"]],
                           "bbox_mode": BoxMode.XYWH_ABS, "category_id": id,}
                    objs.append(obj)
                    break
            if obj is None:
                if unknown_bird_category:
                    obj = {"bbox": [row["x"], row["y"], row["width"], row["height"]],
                           "bbox_mode": BoxMode.XYWH_ABS, "category_id": len(class_names)}
                    objs.append(obj)
                else:
                    continue

        record["annotations"] = objs
        dataset_dicts.append(record)
        
        return dataset_dicts


def register_datasets_AWS(s3client, my_bucket data_dirs, img_ext, birds_species_names, bird_species_colors=None):
    """
    Register dataset as part of Detectron2's dataset and metadataset catalogs
    For each dataset directory to be registered, a "bird-only" and "bird-species" dataset will be registered
    INPUTS:
    data_dirs: list of directories containing dataset images to be registered
    img_ext: file extension for images in dataset
    birds_species_names: names of bird species to be registered. Species not in this list will be registered as
                        "Unknown Bird"
    bird_species_colors: List of colors for corresponding to bird species to be used for visualizations. Color
                         format should be tuple containing RGB values between 0-255 eg. (255,0,0) for red
    """
    
    # get the list of directors for the train and split 
    dd = [x['Key'] for x in s3client.list_objects_v2(Bucket=my_bucket, Prefix=data_dirs) ['Contents']] 
    
    for data_key in dd:
        d = data_key.split('/')[0]

        # birds only section #########################################################################################
        
        # check if we have the bird only in the dataset catalog
        if f"birds_only_{d}" in DatasetCatalog.list():
            DatasetCatalog.remove(f"birds_only_{d}")
        # this function will register the data set into detectron2 package
        DatasetCatalog.register(f"birds_only_{d}", lambda d=d: get_bird_only_dicts_AWS(s3client, my_bucket, data_key, img_ext))
        if f"birds_only_{d}" in MetadataCatalog.list():
            MetadataCatalog.remove(f"birds_only_{d}")
        MetadataCatalog.get(f"birds_only_{d}").set(thing_classes=["Bird"])

        # birds species section #########################################################################################
        if f"birds_species_{d}" in DatasetCatalog.list():
            DatasetCatalog.remove(f"birds_species_{d}")
        DatasetCatalog.register(f"birds_species_{d}", lambda d=d:
                                get_bird_species_dicts_AWS(s3client, my_bucket, data_key,birds_species_names,img_ext='.JEPG',unknown_bird_category=True))
        if f"birds_species_{d}" in MetadataCatalog.list():
            MetadataCatalog.remove(f"birds_species_{d}")
        MetadataCatalog.get(f"birds_species_{d}").set(thing_classes=birds_species_names + ["Unknown Bird"])
        
        if bird_species_colors is not None:
            MetadataCatalog.get(f"birds_species_{d}").set(thing_colors=bird_species_colors + [(0, 0, 0)])

