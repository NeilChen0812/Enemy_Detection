import os
import sys
import numpy as np
import cv2
from PIL import ImageGrab
from mrcnn.config import Config
from datetime import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
FACTION_MODEL_PATH = os.path.join(ROOT_DIR ,"weights/mask_rcnn_faction_4_v1.h5")
if not os.path.exists(FACTION_MODEL_PATH):
    print("Model not found in path:", FACTION_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class FactionsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "factions"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 factions
 
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

config = FactionsConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(FACTION_MODEL_PATH, by_name=True)

# Class in model
class_names = ['BG', 'USA', 'INS']
team = 'USA'

def apply_mask(image, mask, color, alpha=0.5):
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

while True:
    # Capture screen
    img_rgb = ImageGrab.grab()
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    image = cv2.resize(img_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)

    a=datetime.now() 
    # Run detection
    results = model.detect([image], verbose=1)
    b=datetime.now() 
    print("Time: ",(b-a).seconds)
    r = results[0]
    # Display results
    output = display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'])
    output = cv2.resize(output, (1280, 720), interpolation=cv2.INTER_AREA)
    cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("img", 0, 720)
    cv2.imshow('img', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()