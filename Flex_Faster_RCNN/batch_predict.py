import os
import time
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files.faster_rcnn_framework import FasterRCNN
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from draw_box_utils import draw_box


def create_model(num_classes):
    # resNet50+fpn+faster_RCNN
    # norm_layer should be consistent with training.
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

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
    model = create_model(num_classes=21)

    # load train weights
    # fpn weight
    train_weights = "/Users/maojietang/Downloads/fasterrcnn_20220225.pth"

    # one feature map weight
    # train_weights = '/Users/maojietang/Documents/Audubon_F21/Flex_Faster_RCNN/save_weights/resNetFpn-model-2.pth'
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)
    data_transform = transforms.Compose([transforms.ToTensor()])

    # read class_indict
    # label_json_path = '/Users/maojietang/Documents/Audubon_F21/Flex_Faster_RCNN/Birds_classes.json'
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    # save path
    save_path = os.path.join(os.getcwd(), 'test_result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load image
    # image files
    imgs_root = '/Users/maojietang/Downloads/VOCdevkit/Test/Image'
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # only get all jpg images
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    batch_size = 8  # batch size
    model.eval()
    with torch.no_grad():
        # init
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                print('Processing {}'.format(img_path))
                origin_img = Image.open(img_path)
                img = data_transform(origin_img)
                img = torch.unsqueeze(img, dim=0)

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
                print('{} objects found in {}'.format(len(predict_scores),
                                                      img_path))
                draw_box(origin_img,
                         predict_boxes,
                         predict_classes,
                         predict_scores,
                         category_index,
                         thresh=0.50,
                         line_thickness=3)
                # plt.imshow(origin_img)
                # plt.show()

                # # save result
                annotated_img_path = img_path.split('.')[0]
                annotated_img = ''.join([annotated_img_path.split('/')[-1], '_annotated.jpg'])
                annotated_path = os.path.join(save_path, annotated_img)
                origin_img.save(annotated_path)


if __name__ == '__main__':
    main()
