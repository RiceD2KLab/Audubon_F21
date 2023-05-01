import torch
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
import sqlconnection as sql
import time
import cv2

# Image.MAX_IMAGE_PIXELS = None

'''
Gets and returns the dimensions of a chosen image
'''
def getImageSize(path):
    width, height = Image.open(path).size
    return height, width

'''
The code to run just the detector on a selected image
'''
def detect_birds(path):

    print("path: " + path)
    #############################################################################
    # Inputs
    detector_path = r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\models\bird_only.pth'
    # classifier_path = "data/models/classifier.pth"
    img_path = path

    #############################################################################
    # Setup

    torch.manual_seed(2023)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = 23

    #############################################################################
    # Detection

    # Load bird detector
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

    # Have a table of coornidates of the bounding boxes
    boxes_array = boxes[0]['boxes'].detach().cpu().numpy()
    boxes_df = pd.DataFrame(boxes_array, columns=['x1', 'y1', 'x2', 'y2'])

    # Extract the bounding boxes from the image
    cropped_birds = []
    cropped_birds_expanded = []
    height, width = getImageSize(path)

    sql.createNewTable()
    time.sleep(5)
    print("Table created?")

    for box in boxes_array:
        x1, y1, x2, y2 = box
        cropped_birds.append(image.crop((x1, y1, x2, y2)))
        
        sql.addRow(x1, y1, x2-x1, y2-y1)
        cropped_birds_expanded.append(image)
        #cropped_birds_expanded.append(draw_boxv2(path, int(x1), int(y1), int(x2), int(y2), height, width))

    for i in range(len(cropped_birds)):
        cropped_birds[i].save(r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\upload\bird' + str(i) + ".jpg")
        cropped_birds_expanded[i].save(r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\upload\expanded_bird' + str(i) + ".jpg")
        #cv2.imwrite(r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\upload\expanded_bird' + str(i) + ".jpg", cropped_birds_expanded[i])
    return cropped_birds

'''
The code to run both the detector and the classifier on a selected image

Returns an array with the labels of the birds found in the image
'''
def bird_classifier(path):
    torch.manual_seed(2023)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    detector_path = r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\models\bird_only.pth'
    classifier_path = r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\models\bird_classifier.pth'
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

    for box in boxes_array:
        x1, y1, x2, y2 = box
        cropped_birds.append(image.crop((x1, y1, x2, y2)))
        
        #sql.addRow(x1, y1, x2-x1, y2-y1)

        cropped_birds_expanded.append(draw_box(path, int(x1), int(y1), int(x2), int(y2)))

    for i in range(len(cropped_birds)):
        cropped_birds[i].save(r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\upload\bird' + str(i) + ".jpg")
        cv2.imwrite(r'C:\Users\dosjo\Documents\COMP 449\Audubon_F21\code\server\upload\expanded_bird' + str(i) + ".jpg", cropped_birds_expanded[i])
   
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
                'Snowy Egret SNEGA', 'Tri-Colored Heron Adult ', 'Tricolored Heron Adult TRHEA', 'White Ibis Adult WHIBA', 'White Ibis Chick WHIBC', 
                'White Morph Adult MEGRT', 'White Morph Reddish Egret Adult REEGWMA']
   
    '''fig, axes = plt.subplots(1, 3, figsize=(8, 5))
    for idx, image in enumerate(cropped_birds):
        axes[idx].imshow(image)
        axes[idx].set_title(class_names[boxes_df['label'].iloc[idx]])
        axes[idx].axis('off')'''

    label_names = []
    for num in labels:
        label_names.append(class_names[num])
    return label_names

'''
Draws a box around the bird that we are currently working on and returns it to be saved
'''
def draw_box(path, x1, y1, x2, y2):
    img = cv2.imread(path)
    imgCopy = img.copy()
    cv2.rectangle(imgCopy, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return imgCopy