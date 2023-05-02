import torch
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import cv2
import os

# Image.MAX_IMAGE_PIXELS = None

dirname = os.path.dirname(__file__)

'''
    Gets and returns the dimensions of a chosen image
    Inputs: 
            path - denotes the absolute path to the image that needs to be checked
    Returns: 
            Integers denoting the height and width of the image in pixels
'''
def getImageSize(path):
    width, height = Image.open(path).size
    return height, width

'''
    The code to run both the detector and the classifier on a selected image
    Inputs: 
            path - denotes the absolute path to the image that needs to be checked
    Returns: 
            Two arrays, first of which just has the names of the birds and second of which will be used to generate a csv file
'''
def bird_classifier(path):
    print(dirname)
    torch.manual_seed(2023)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    detector_path = os.path.join(dirname, 'models/bird_only.pth')
    classifier_path = os.path.join(dirname, 'models/bird_classifier.pth')
    img_path = path
    num_class = 23
   
    # Load trained bird detector
    detector = torch.load(detector_path, map_location=device)
    detector.eval()
   
    # Upload and transform image 
    transformer = transforms.Compose([transforms.PILToTensor(),
                                    transforms.ConvertImageDtype(torch.float)])
   
    image = Image.open(img_path).convert('RGB')
    image_tensor = transformer(image)
    image_tensor = image_tensor.unsqueeze_(0)  # So the image is treated as a batch 
    image_tensor = image_tensor.to(device)
   
    # Detect birds in image
    boxes = detector(image_tensor)
    print(boxes)
   
    # Have a table of coornidates of the bounding boxes
    boxes_array = boxes[0]['boxes'].detach().cpu().numpy()
    boxes_df = pd.DataFrame(boxes_array, columns=['x1', 'y1', 'x2', 'y2'])
   
    # Extract the bounding boxes from the image
    cropped_birds = []
    cropped_birds_expanded = []
    height, width = getImageSize(path)
    
    bird_data = []

    for box in boxes_array:
        x1, y1, x2, y2 = box
        cropped_birds.append(image.crop((x1, y1, x2, y2)))
        
        #sql.addRow(x1, y1, x2-x1, y2-y1)
        bird_data.append(['', '', int(x1), int(y1), int(x2-x1), int(y2-y1)])

        cropped_birds_expanded.append(draw_box(path, int(x1), int(y1), int(x2), int(y2)))

    for i in range(len(cropped_birds)):
        string1 = 'upload/bird' + str(i) + '.jpg'
        string2 = 'upload/expanded_bird' + str(i) + '.jpg'
        cropped_birds[i].save(os.path.join(dirname, string1))
        cv2.imwrite(os.path.join(dirname, string2), cropped_birds_expanded[i])
   
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
   
    resnet = torch.load(classifier_path, map_location=device)
    resnet.eval()
                                                
    # Classify birds
    bird_tensors = torch.stack([preprocess(bird_image) for bird_image in cropped_birds])
    bird_tensors = bird_tensors.to(device)
    label_scores = resnet(bird_tensors)
    labels = []
    scores = []
    for label_score in label_scores:
        label = label_score.argmax().item()
        score = max(label_score.detach().softmax(dim=0)).item()
        labels.append(label)
        scores.append(score)
   
    # Add labels and scores to the table
    boxes_df['label'] = labels
    boxes_df['score'] = scores
    print(boxes_df)
    print(f"Number of birds detected: {len(boxes_df)}")
   
    class_names = ['Black Skimmer Adult BLSKA', 'Black-Crowned Night Heron Adult BCNHA', 'Brown Pelican Adult BRPEA', 
                'Brown Pelican Chick BRPEC', 'Brown Pelican Juvenile BRPEJ', 'Cattle Egret Adult CAEGA', 'Great Blue Heron Adult GBHEA', 
                'Great Blue Heron Chick GBHEC', 'Great Blue Heron Juvenile GBHEJ', 'Great Egret Adult GREGA', 'Great Egret Chick GREGC', 
                'Laughing Gull Adult LAGUA', 'Mixed Tern Adult MTRNA', 'Other Bird OTHRA', 'Reddish Egret Adult REEGA', 'Roseate Spoonbill Adult ROSPA', 
                'Snowy Egret SNEGA', 'Tri-Colored Heron Adult TRHEA', 'Tricolored Heron Adult TRHEA', 'White Ibis Adult WHIBA', 'White Ibis Chick WHIBC', 
                'White Morph Adult MEGRT', 'White Morph Reddish Egret Adult REEGWMA']
   
    '''
    fig, axes = plt.subplots(1, 3, figsize=(8, 5))
    for idx, image in enumerate(cropped_birds):
        axes[idx].imshow(image)
        axes[idx].set_title(class_names[boxes_df['label'].iloc[idx]])
        axes[idx].axis('off')
    '''

    label_names = []

    for i in range(len(labels)):
        name = class_names[labels[i]]
        arr = name.split()
        bird_id = arr.pop()
        bird_name = " ".join(arr)

        label_names.append(name)
        bird_data[i][0] = bird_id
        bird_data[i][1] = bird_name

    bird_data.insert(0, ["class_id", "desc", "x", "y", "width", "height"])

    return label_names, bird_data

'''
    Draws a box around the bird that we are currently working on and returns it to be saved
    Inputs: 
            path - denotes the absolute path to the image being worked on
            x1 - the x coordinate of the top left corner of the box
            y1 - the y coordinate of the top left corner of the box 
            x2 - the x coordinate of the bottom right corner of the box
            y2 - the y coordinate of the bottom right corner of the box

    Returns:
            A copy of the image found in path with a red box drawn around it
'''
def draw_box(path, x1, y1, x2, y2):
    img = cv2.imread(path)
    imgCopy = img.copy()
    cv2.rectangle(imgCopy, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return imgCopy