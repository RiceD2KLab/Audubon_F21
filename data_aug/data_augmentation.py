def crop_minor(csv_file, crop_height, crop_width, output_dir, minor_species, annot_file_ext='bbx', file_dict={}):
  file_name = os.path.split(csv_file)[-1][:-4]

  annot_dict = csv_to_dict(csv_path = csv_file, annot_file_ext=annot_file_ext)
  annotation_lst = [list(x.values()) for x in annot_dict['bbox']]

  image_file = csv_file.replace(annot_file_ext, 'JPG')
  assert os.path.exists(image_file)

  #Load the image
  image = Image.open(image_file)

  minors = []
  for dic in annot_dict['bbox']:
    if dic["desc"] in minor_species:
      minors.append(dic)
      
  for i in range(len(minors)):
    minor = minors[i]
    left, upper, right, lower = minor["xmin"], minor["ymax"], minor["xmax"], minor["ymin"]
    center_w, center_h = (left + right) // 2, (upper + lower) // 2

    # image.crop((left,upper,right,lower))
    cropped = image.crop((center_w-0.5*crop_width, center_h-0.5*crop_width, center_w+0.5*crop_width, center_h+0.5*crop_height)) 
    cropped.save(output_dir+"/"+file_name+"_"+str(i+1).zfill(2)+ ".JPG")


def dataset_aug(input_dir, output_dir, minor_species, annot_file_ext = 'bbx', crop_height = 640, crop_width = 640):
  if annot_file_ext == 'bbx':
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file[-3:] == 'bbx']
  for file in tqdm(files, desc='Cropping files'):
        crop_minor(csv_file = file,crop_height = crop_height,crop_width = crop_width, output_dir = output_dir,minor_species = minor_species)
