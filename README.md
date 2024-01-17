# Enemy Detecttion Base on Mask R-CNN

## Introduction

This is a project for enemy detection base on Mask R-CNN. The project is based on [ahmedfgad/Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2) which was forked from [Matterport's implementation of Mask R-CNN](https://github.com/matterport/Mask_RCNN)

## Environment

- OS: Windows 11
- GPU: RTX 3050
- CUDA: 11.2
- CUDNN: 8.1.1
- Python: 3.7.9
- Tensorflow-GPU: 2.5.0

## Installation

1. Clone this repository
   2 Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
2. Run setup from the repository root directory
   ```bash
   python3 setup.py install
   ```
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
4. Download the trained weights and test video from [here](https://drive.google.com/drive/folders/1ClBylrY6dv0jFc3NeP70FSuQR3v2nBUz?usp=sharing)

## Usage

### Detection:

1. Put the .h5 weights files in the folder `weights`
2. Run the following command to
   - Real-time detect on screen
     ```bash
     python3 realtime[version].py
     ```
   - Detect on video
     ```bash
      python3 video[version].py
     ```

### Training:

1. Put the .h5 weights files in the folder `weights`
2. Put the training data in the folder `data\dataset\[dataset_name]`
3. Run the following command to train the model
   ```bash
   python3 train[version].py
   ```

### Build the dataset:

1. Use `Labelme` to label the images
2. Create the following folders in the folder `data\classify`
   - cv2_mask
   - json
   - labelme_json
   - pic
3. Put the output json files in the folder `data\classify\json`
4. Run the following command in `labelme` environment to convert the dataset
   ```bash
   python3 json2dataset.py
   ```
5. Run the following command to classify the dataset
   ```bash
   python3 classify.py
   ```
6. Copy the following folders to the folder `data\dataset\[dataset_name]`
   - cv2_mask
   - json
   - labelme_json
   - pic

### Generate the dataset:

1. Film videos only contains one class of object and put them in the folder `data\classify` with the name of the class (e.g. `Class1.mp4`)
2. Repeat the following steps for each class of object

   1. Run the following command to generate the dataset
      ```bash
      python3 generate_data.py
      ```
   2. Run the following command to classify the dataset
      ```bash
      python3 classify.py
      ```
   3. Set the label in `mask2json.py` and run the following command to convert the dataset to labelme format
      ```bash
      python3 mask2json.py
      ```
   4. Clear the following folders and repeat the above steps for the next class of object
      - cv2_mask
      - labelme_json
      - pic

3. Run the following command in `labelme` environment to convert the dataset
   ```bash
   python3 json2dataset.py
   ```
4. Run the following command to classify the dataset
   ```bash
   python3 classify.py
   ```
5. Copy the following folders to the folder `data\dataset\[dataset_name]`
   - cv2_mask
   - json
   - labelme_json
   - pic
