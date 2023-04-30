from config import PLOTS_PATH, DESC_MAPPING
from config import OLD_CSV_PATH, NEW_CSV_PATH, TILED_OLD_CSV_PATH, TILED_NEW_CSV_PATH
from config import DATA_PATH, IMG_PATH, CROPPED_PATH
from src.data.convert_annotations import write_csv, add_class_id_and_data_exploration
from src.data.crop_birds import cropping


write_csv(OLD_CSV_PATH, NEW_CSV_PATH, DESC_MAPPING)
write_csv(TILED_OLD_CSV_PATH, TILED_NEW_CSV_PATH, DESC_MAPPING)

add_class_id_and_data_exploration(NEW_CSV_PATH,
                                  'histogram of bird species (original images)',
                                  DATA_PATH, PLOTS_PATH)
add_class_id_and_data_exploration(TILED_NEW_CSV_PATH,
                                  'histogram of bird species (tiled images)',
                                  DATA_PATH, PLOTS_PATH)

cropping(NEW_CSV_PATH, IMG_PATH, CROPPED_PATH)