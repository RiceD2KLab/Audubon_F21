def flip_img(img, info_dict, output_path):
  name = ("_flipped.").join(info_dict["file_name"].split("."))

  transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
  flipped = transform(img)
  flipped.save(output_dir+"/"+name)


  img_height, img_width, img_depth = info_dict['img_size']

  instance_dict = info_dict["bbox"][0]
  instancef_dict = {}
  instancef_dict['class'] = instance_dict['class']
  instancef_dict['desc'] = instance_dict['desc']
  instancef_dict['xmin'] = max(img_width - instance_dict['xmax'], 0)               # Horizontal Flip
  instancef_dict['ymin'] = instance_dict['ymin']
  instancef_dict['xmax'] = min(img_width - instance_dict['xmin'], img_width)       # Horizontal Flip
  instancef_dict['ymax'] = instance_dict['ymax']

  flipped_dict = {}
  flipped_dict["bbox"] = [instancef_dict]
  flipped_dict["file_name"] = name
  flipped_dict["img_size"] = info_dict["img_size"]

  dict_to_csv(flipped_dict, empty=False, output_path=output_dir)
  
  
  
  
def aug_minor(csv_file, crop_height, crop_width, output_dir, minor_species, annot_file_ext='bbx'):
  file_name = os.path.split(csv_file)[-1][:-4]

  annot_dict = csv_to_dict(csv_path = csv_file, annot_file_ext=annot_file_ext)
  annotation_lst = [list(x.values()) for x in annot_dict['bbox']]

  image_file = csv_file.replace(annot_file_ext, 'JPG')
  assert os.path.exists(image_file)

  #Load the image
  image = Image.open(image_file)
  width, height = image.size

  minors = []
  for dic in annot_dict['bbox']:
    if dic["desc"] in minor_species:
      minors.append(dic)
      
  for i in range(len(minors)):
    minor = minors[i]
    center_w, center_h = (minor["xmin"] + minor["xmax"]) // 2, (minor["ymin"] + minor["ymax"]) // 2

    
    left, top, right, bottom = center_w-0.5*crop_width, center_h-0.5*crop_width, center_w+0.5*crop_width, center_h+0.5*crop_height
    if left < 0:
      left, right = 0, crop_width
    if right > width:
      left, right = width - crop_width, width
    if top < 0:
      top, bottom = 0, crop_height
    if bottom > height:
      top, bottom = height - crop_height, height
    
    cropped = image.crop((left, top, right, bottom)) 
    cropped.save(output_dir+"/"+file_name+"_"+str(i+1).zfill(2)+ ".JPG")

    instance_dict = {}
    instance_dict['class'] = minor['class']
    instance_dict['desc'] = minor['desc']
    instance_dict['xmin'] = max(minor['xmin'] - left, 0)
    instance_dict['ymin'] = max(minor['ymin'] - top, 0)
    instance_dict['xmax'] = min(minor['xmax'] - left, crop_width)
    instance_dict['ymax'] = min(minor['ymax'] - top, crop_height)

    minor_dict = {}
    minor_dict["bbox"] = [instance_dict]
    minor_dict["file_name"] = file_name+"_"+str(i+1).zfill(2)+ ".JPG"
    minor_dict["img_size"] = (crop_width,crop_height,3)                     
    
    dict_to_csv(minor_dict, empty=False, output_path=output_dir)

    flip_img(img = cropped, info_dict = minor_dict, output_path = output_dir)


def dataset_aug(input_dir, output_dir, minor_species, annot_file_ext = 'bbx', crop_height = 640, crop_width = 640):

  if annot_file_ext == 'bbx':
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file[-3:] == 'bbx']
  for file in tqdm(files, desc='Cropping files'):
    aug_minor(csv_file = file,crop_height = crop_height,crop_width = crop_width, output_dir = output_dir,minor_species = minor_species)

