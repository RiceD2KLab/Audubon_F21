import matplotlib.pyplot as plt
import csv
import os
import shutil
import zipfile

output_dir = 'cropped_birds_dir'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def crop_to_bounding_boxes(dataset, output_dir):
  """
    Crop images in a dataset based on their bounding box annotations and save the cropped images and annotations to a zip file.

    Input:
      dataset (dict): A dictionary containing the paths to the images and corresponding bounding box annotation CSV files. 
                      The dictionary should have the keys 'jpg' and 'csv', which should contain lists of file paths to the 
                      image and CSV files, respectively.
      output_dir (str): The directory where the resulting zip file should be saved.

    Output:
      zip_path (str): The path to the resulting zip file containing the cropped images and annotations.
  """
    # Create a temporary directory to store the cropped images and annotations
    temp_dir = 'temp_dir'
    os.makedirs(temp_dir, exist_ok=True)

    # Loop over the images in the dataset
    output = []
    for i, (jpg_path, csv_path) in enumerate(zip(dataset['jpg'], dataset['csv'])):
        # Load the image and bounding box data
        img = plt.imread(jpg_path)
        with open(csv_path) as f:
            csv_reader = csv.DictReader(open(csv_path))
            bbx = [(row['class_id'], row['desc'], int(row['x']), int(row['y']), int(row['width']), int(row['height']),) for row in csv_reader]

        # Crop the image to each bounding box and save the cropped images and annotations
        for j, box in enumerate(bbx):
            # Calculate the dimensions of the square crop as the maximum of the bounding box h/w
            class_id, desc, x, y, width, height = box
            crop_dim = max(width, height)
            # Center the crop on the bounding box and crop
            x_ = max(0, x + width // 2 - crop_dim // 2)
            y_ = max(0, y + height // 2 - crop_dim // 2)
            cropped_img = img[y_:y_+crop_dim, x_:x_+crop_dim]
            cropped_path = os.path.join(temp_dir, f'{i}_{j}.jpg')
            plt.imsave(cropped_path, cropped_img)
            annotation_path = os.path.join(temp_dir, f'{i}_{j}.csv')
            with open(annotation_path, 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['class_id', 'desc', 'x', 'y', 'width', 'height'])
                csv_writer.writerow([class_id, desc, 0, 0, crop_dim, crop_dim])

            # Append the cropped image and annotation data to the output list
            output.append({'image': cropped_path, 'annotation': annotation_path, 'class_id': class_id, 'class_name': desc})

    # Create a zip file containing the cropped images and annotations
    zip_path = os.path.join(output_dir, 'cropped_birds.zip')
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for item in output:
            zip_file.write(item['image'])
            zip_file.write(item['annotation'])

    # Remove the temporary directory and return the path to the zip file
    shutil.rmtree(temp_dir)
    return zip_path
