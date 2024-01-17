import os
import numpy as np
import cv2
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import yaml
from PIL import Image
 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
iter_num = 0

# Local path to trained weights file
PEOPLE_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_people_v4.h5")
if not os.path.exists(PEOPLE_MODEL_PATH):
    print("Model not found in path:", PEOPLE_MODEL_PATH)

class FactionsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "factions"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 factions
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 50

config = FactionsConfig()
config.display()

class DrugDataset(utils.Dataset):
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.safe_load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        # Add classes
        self.add_class("shapes", 1, "BLUFOR")
        self.add_class("shapes", 2, "REDFOR")
 
        for i in range(count):
            print(i)
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            print(type(cv_img))
 
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
 
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []

        for i in range(len(labels)):
            if labels[i].find("BLUFOR") != -1:
                labels_form.append("BLUFOR")
            elif labels[i].find("REDFOR") != -1:
                labels_form.append("REDFOR")


        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


dataset_root_path = "data/dataset/faction_2/"   # dataset root path
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "cv2_mask"
imglist = os.listdir(img_floder)
count = len(imglist)

dataset_train = DrugDataset()
dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_train.prepare()
 
dataset_val = DrugDataset()
dataset_val.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_val.prepare()
 
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
 
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last
 
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    # print(PEOPLE_MODEL_PATH)
    model.load_weights(PEOPLE_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
 
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10,
            layers="all")
