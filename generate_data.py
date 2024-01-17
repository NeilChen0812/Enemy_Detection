import os
import sys
import numpy as np
import cv2
import yaml
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
SHAPE_MODEL_PATH = os.path.join(ROOT_DIR ,"weights/mask_rcnn_people_v4.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(SHAPE_MODEL_PATH):
    print("Model not found in path:", SHAPE_MODEL_PATH)
    # utils.download_trained_weights(SHAPE_MODEL_PATH)

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
    IMAGES_PER_GPU = 3
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes
 
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

label = 'INS'
video_path = f'data/classify/{label}.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    # Customize
    # file_names = next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    ret, frame = cap.read()
    if not ret:
        break

    w, h = 1080, 1080
    x, y = 440, 0
    frame = cv2.resize(frame[y:y+h, x:x+w], (1024, 1024))
    a=datetime.now()
    # Run detection
    results = model.detect([frame], verbose=1)
    b=datetime.now()
    # Visualize result
    print("Time",(b-a).seconds)
    r = results[0]
    mask = np.zeros((1024, 1024))
    for i in range(r['rois'].shape[0]):
        mask = np.logical_or(mask, r['masks'][:, :, i]).astype(np.uint8)

    if np.all(mask == 0):
        continue

    # 建立資料夾
    output_folder = './data/classify/labelme_json/'
    folder_path = os.path.join(output_folder, f'{label}{frame_count}_json')
    os.makedirs(folder_path, exist_ok=True)

    # 儲存處理後的每一幀
    img_path = os.path.join(folder_path, 'img.png')
    cv2.imwrite(img_path, frame)
    mask_path = os.path.join(folder_path, 'label.png')
    # cv2.imwrite(mask_path, cv2.merge([np.zeros_like(mask), np.zeros_like(mask), mask*128]))
    cv2.imwrite(mask_path, mask)
    print(f'Finish {frame_count} frame')
    print('Save to', folder_path)

    frame_count += 1