from samples.coco import coco
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from PIL import ImageGrab
from mrcnn.config import Config
from datetime import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

# import train_tongue
# class InferenceConfig(coco.CocoConfig):


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names, scores=None):
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    N = boxes.shape[0]

    colors = colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]

    for i, c in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = "{} {:.3f}".format(label, score) if score else label

        # Mask
        mask = masks[:, :, i]
        if label == 'person':
            image = apply_mask(image, mask, (0, 1, 1), alpha=0.5)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
        # image = apply_mask(image, mask, c)
        # image = cv2.rectangle(image, (x1, y1), (x2, y2), c, 2)
        # image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, c, 2)
    return image


i = 0
while True:
    # file_names = next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    img_rgb = ImageGrab.grab()
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    w, h = 1080, 1080
    x, y = 440, 0
    # image = cv2.resize(img_bgr[y:y+h, x:x+w], (720, 720), interpolation=cv2.INTER_AREA)
    image = cv2.resize(img_bgr, (1280, 720), interpolation=cv2.INTER_AREA)

    a = datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    b = datetime.now()
    # Visualize resultsq
    print("Time", (b-a).seconds)
    r = results[0]
    output = display_instances(image, r['rois'], r['masks'], r['class_ids'],
                               class_names, r['scores'])
    output = cv2.resize(output, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("img", 0, 720)
    cv2.imshow('img', output)
    # cv2.imwrite('output/' + str(i) + '.jpg', output)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
