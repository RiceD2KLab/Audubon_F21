import os
import time
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
from network_files.faster_rcnn_framework import FasterRCNN, AnchorsGenerator
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from draw_box_utils import draw_box
from backbone.mobile_net_v2 import MobileNetV2


def create_model(num_classes):
    # resNet50+fpn+faster_RCNN
    # norm_layer should be consistent with training.
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_score_thresh=0.5,
                       image_mean=[0.4833599, 0.46734715, 0.39651462],
                       image_std=[0.17723781, 0.16923204, 0.1732905]
                       )


    # one feature map model
    # backbone = MobileNetV2(norm_layer=torch.nn.BatchNorm2d).features
    # backbone.out_channels = 1280  # num_classes
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))

    #

    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=7)

    # load train weights
    train_weights = "C://Users//VelocityUser//Documents//Audubon_F21//Flex_Faster_RCNN//save_weights//resNetFpn-model-80.pth"

    # train_weights = "/Users/maojietang/Downloads/mobile-model-10.pth"

    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    # label_json_path = '/Users/maojietang/Documents/Audubon_F21/Flex_Faster_RCNN/Birds_classes.json'

    label_json_path = 'C://Users//VelocityUser//Documents//Audubon_F21//Flex_Faster_RCNN//helper//bird_class.json'

    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    # load image

    original_img = Image.open("C://Users//VelocityUser//Documents//D2K TDS A//6_class_combine\images//102741 00001.JPG")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("No object!")
        print('{} objects we hanve found'.format(len(predict_boxes)))
        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.5,
                 line_thickness=3)
        plt.imshow(original_img)
        plt.show()
        # save result
        # original_img.save("test_result.jpg")


if __name__ == '__main__':
    main()

