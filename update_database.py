import splitfolders
from config import PLOTS_PATH, DESC_MAPPING
from config import OLD_CSV_PATH, NEW_CSV_PATH, TILED_OLD_CSV_PATH, TILED_NEW_CSV_PATH
from config import DATA_PATH, IMG_PATH, CROPPED_PATH, CROPPED_SPLIT_PATH, SEED
from src.data.convert_annotations import write_csv, add_class_id_and_data_exploration
from src.data.crop_birds import cropping


def update_database():
    ''' 
    Process the dataset using the following steps: 
        1. Update annotations on the original images by writing a new CSV file NEW_CSV_PATH.
        2. Update annotations on the tiled images by creating a new CSV file TILED_NEW_CSV_PATH.
        3. Plot a histogram of bird species distribution in the full image dataset.
        4. Plot a histogram of bird species distribution in the tiled image dataset.
        5. Crop the birds from the original images into folders according to their species class, 
           using annotations in the NEW_CSV_PATH file and saving the cropped images in CROPPED_PATH.
        6. Split the cropped images into train, validation, and test sets, and save them 
           in a separate directory at CROPPED_SPLIT_PATH, with a ratio of (0.8, 0.1, 0.1) respectively.
    
    Output: 
        Updated annotations in NEW_CSV_PATH and TILED_NEW_CSV_PATH
        Histograms of species class distribution in the full image and tiled image datasets
        Cropped bird images organized into folders by species class
        Training, validation and test sets of cropped images
    '''

    # update annotations on original images
    write_csv(OLD_CSV_PATH, NEW_CSV_PATH, DESC_MAPPING)

    # update annotations on tiled images
    write_csv(TILED_OLD_CSV_PATH, TILED_NEW_CSV_PATH, DESC_MAPPING)

    # add class id and data exploration on updated annotations of original images
    add_class_id_and_data_exploration(NEW_CSV_PATH,
                                      'Histogram of bird species (original images)',
                                      DATA_PATH, PLOTS_PATH)

    # add class id and data exploration on updated annotations of tiled images
    add_class_id_and_data_exploration(TILED_NEW_CSV_PATH,
                                      'Histogram of bird species (tiled images)',
                                      DATA_PATH, PLOTS_PATH)

    # croppe birds from original images into folders according to their species
    cropping(NEW_CSV_PATH, IMG_PATH, CROPPED_PATH)

    # split cropped images into train, val, and test sets and save them in a separate directory
    splitfolders.ratio(CROPPED_PATH, output=CROPPED_SPLIT_PATH,
                       seed=SEED,
                       ratio=(0.8, 0.1, 0.1))


if __name__ == '__main__':
    update_database()
