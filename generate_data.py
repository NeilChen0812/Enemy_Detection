import os
import sys
import numpy as np
import cv2
from mrcnn.config import Config
from datetime import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
PEOPLE_MODEL_PATH = os.path.join(ROOT_DIR ,"weights/mask_rcnn_people_v4.h5")
if not os.path.exists(PEOPLE_MODEL_PATH):
    print("Model not found in path:", PEOPLE_MODEL_PATH)

class PeopleConfig(Config):
    # Give the configuration a recognizable name
    NAME = "people"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1     # background + 1 class
 
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)    # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE =100
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

class InferenceConfig(PeopleConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(PEOPLE_MODEL_PATH, by_name=True)

# Class in video
label = 'INS'
# Input video path
video_path = f'data/classify/{label}.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Resize frame
    w, h = 1080, 1080
    x, y = 440, 0
    frame = cv2.resize(frame[y:y+h, x:x+w], (1024, 1024))
    a=datetime.now()
    # Run detection
    results = model.detect([frame], verbose=1)
    b=datetime.now()
    print("Time",(b-a).seconds)
    r = results[0]
    mask = np.zeros((1024, 1024))
    for i in range(r['rois'].shape[0]):
        mask = np.logical_or(mask, r['masks'][:, :, i]).astype(np.uint8)
    
    # Skip if no people in frame
    if np.all(mask == 0):
        continue

    # Make folder to save data
    output_folder = './data/classify/labelme_json/'
    folder_path = os.path.join(output_folder, f'{label}{frame_count}_json')
    os.makedirs(folder_path, exist_ok=True)

    # Save data
    img_path = os.path.join(folder_path, 'img.png')
    cv2.imwrite(img_path, frame)
    mask_path = os.path.join(folder_path, 'label.png')
    cv2.imwrite(mask_path, mask)

    print(f'Finish {frame_count} frame')
    print('Save to', folder_path)
    frame_count += 1