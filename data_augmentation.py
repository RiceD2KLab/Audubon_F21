# Import useful libraries
import os, glob
import pandas as pd

from utils.augmentation import dataset_aug

# data_dir is the path that contains both images and annotations (image: jpg; annotation: csv or bbx)
input_dir = 'C://Users\\VelocityUser\\Documents\\0224_data'
# output dir is the path where you want to output new files. Please use the folder you defined above.
#mkdir -p './data/cropped'
os.makedirs('C://Users\\VelocityUser\\Documents\\Audubon_F21\\augment_0224_3', exist_ok= True)

output_dir = 'C://Users\\VelocityUser\\Documents\\Audubon_F21\\augment_0224_3'

#!mkdir -p '/content/drive/My Drive/Audubon/aug_data'
#output_dir ='/content/drive/My Drive/Audubon/aug_data'

# Tile size
crop_height = crop_width = 640

# Minimum portion of a bounding box being accepted in a subimage
overlap = 0.2

# List of species that we want to augment
# minor_species = ["Brown Pelican Adult", "Tricolored Heron Adult", "Great Blue Heron Adult"]
minor_species = ["Brown Pelican Adult", "Tricolored Heron Adult"]

# Threshold of non-minor creatures existing in a subimage
thres = .1
dataset_aug(input_dir, output_dir, minor_species, overlap, thres)

#######################################################################################################################
minor_species = ["Great Blue Heron Adult"]
# Threshold of non-minor creatures existing in a subimage
thres = .7
dataset_aug(input_dir, output_dir, minor_species, overlap, thres)

# Check output
print (len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))]))

# shutil.rmtree(output_dir)


target_data = []
for f in glob.glob(os.path.join(output_dir, '*.csv')):
    target_data.append(pd.read_csv(f, header=0,
                                   names=["class_id", "class_name", "x", "y", "width", "height"]))
target_data = pd.concat(target_data, axis=0, ignore_index=True)

# Visualize dataset
# print(f'\n {d} - Bird Species Distribution')
print(target_data["class_name"].value_counts())
print('\n')