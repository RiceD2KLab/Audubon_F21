import cv2
import os
from src.data.utils import csv_to_df, get_file_names
from config import NEW_CSV_PATH, IMG_PATH, CROPPED_PATH


def cropping(csv_path, img_path, cropped_path):
    ''' Crop images based on bounding boxes in csv files and save them in a new folder. '''
    jpg_files = get_file_names(img_path, 'jpg')
    csv_files = get_file_names(csv_path, 'csv')
    for idx in range(len(jpg_files)):
        img = cv2.imread(jpg_files[idx])
        boxes = csv_to_df(csv_files[idx])
        for dummy, box in boxes.iterrows():
            xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']
            class_name = box['class_name']
            cropped_img = img[ymin:ymax, xmin:xmax]
            class_folder = cropped_path + class_name
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            file_name = f"{class_folder}/{idx}_{xmin}_{ymin}.jpg"
            cv2.imwrite(file_name, cropped_img)


cropping(NEW_CSV_PATH, IMG_PATH, CROPPED_PATH)
