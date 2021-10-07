# train_net.py 
#
# Train object detection model on Houston Audubon dataset
#
# Authors: Krish Kabra, Minxuan Luo, Alexander Xiong, William Lu
# Copyright (C) 2021-2022 Houston Audubon and others
#
# WORK IN PROGRESS:
# 1. Update how argparser is currently working within setup, train, and eval functions 
# 2. Fix detectron2 crash with uncropped images. 

import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

def get_bird_dicts(data_dir,img_ext='.JPG',dir_exceptions=[]): 
  """
  Format dataset to detectron2 standard format. 
  INPUTS: 
    data_dir -- directory containing datatset folders
    img_ext -- file extension for images in dataset
    dir_exceptions -- folders to ignore in dataset directory 
  OUTPUTS: 
    dataset_dicts -- list of dictionaries in detectron2 standard format
  """ 
  dirs = [d for d in os.listdir(data_dir) if d not in dir_exceptions]

  dataset_dicts = []
  idx = 0
  for d in dirs:
    for file_csv in glob.glob(os.path.join(data_dir,d,'*.csv')): 
      
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
                  
      idx += 1  

  return dataset_dicts


def setup(args): 
    # Create detectron2 config and predictor 
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")

    data_dir = 'data/'
    img_ext='.JPEG'
    dir_exceptions = ["Annotations 20210912","UAV Image - XML - CSV - Ref"]

    if "birds_uav" in DatasetCatalog.list():
        DatasetCatalog.remove("birds_uav")
    DatasetCatalog.register("birds_uav", lambda d=d: get_bird_dicts(data_dir,img_ext,dir_exceptions))
    if "birds_uav" in MetadataCatalog.list():
        MetadataCatalog.remove("birds_uav")
    MetadataCatalog.get("birds_uav").set(thing_classes=["bird"])

    dataset_dicts = get_bird_dicts(data_dir,img_ext,dir_exceptions)
    for i,d in enumerate(random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("birds_uav"), scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow(f"example image {i}", out.get_image()[:, :, ::-1])
        cv2.waitKey(1)

    return cfg


def train(cfg, dataset_dicts, args): 
    # setup training logger
    setup_logger()
    
    cfg.DATASETS.TRAIN = ("birds_uav",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (bird). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    return trainer.train() 

def eval(cfg,args): 
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    for i,d in enumerate(random.sample(dataset_dicts, 3)):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = outputs["instances"].to("cpu")
        outputs = outputs[outputs.scores > 0.8]
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("birds_uav"), 
                       scale=0.5, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs)
        cv2.imshow(f"eval image {i}", out.get_image()[:, :, ::-1])
        cv2.waitKey(1)

    pass

def main(args): 
    cfg = setup(args)
    train(cfg, args)
    eval(cfg,args)
    cv2.destroyAllWindows()
    pass  

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )