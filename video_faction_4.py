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
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Customize
# Local path to trained weights file
SHAPE_MODEL_PATH = os.path.join(ROOT_DIR ,"weights/mask_rcnn_faction_4_v1.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(SHAPE_MODEL_PATH):
    print("Model not found in path:", SHAPE_MODEL_PATH)
    # utils.download_trained_weights(SHAPE_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Customize
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 1024
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(SHAPE_MODEL_PATH, by_name=True)

# Customize
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'USA', 'INS']
team = 'USA'

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

    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]

    for i, c in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        
        y1, x1, y2, x2 = boxes[i]
        label = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        caption = "{} {:.3f}".format(label, score) if score else label

        # Customize
        # Mask
        mask = masks[:, :, i]
        if label == team:
            image = apply_mask(image, mask, (1, 1, 0), alpha=0.25)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            # image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)
        elif label != "_background_":
            image = apply_mask(image, mask, (0, 0, 1), alpha=0.25)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    return image

def video_detection(path, team):
    total_mask = 0
    error_mask = 0
    frame_count = 0

    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

        a=datetime.now()
        # Run detection
        results = model.detect([image], verbose=1)
        b=datetime.now() 
        # Visualize result
        print("Time: ",(b-a).seconds)
        r = results[0]
        for i in range(r['class_ids'].shape[0]):
            total_mask += 1
            if class_names[r['class_ids'][i]] != team:
                error_mask += 1
        output = display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
        output = cv2.resize(output, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('img', output)
        print('Total Mask:', total_mask)
        print('Error Mask:', error_mask)
        print(f'Finish {frame_count} frame')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
    return total_mask, error_mask
    
USA_video_path = 'videos/USA.mp4'
INS_video_path = 'videos/INS.mp4'
USA_result = video_detection(USA_video_path, 'USA')
INS_result = video_detection(INS_video_path, 'INS')

print('USA (Total Mask, Error Mask): ', USA_result)
print('INS (Total Mask, Error Mask): ', INS_result)

