# Liveness Detector Dataset and Model Training

This project is a complete pipeline for detecting liveness in images using YOLO. It includes custom dataset generation, dataset splitting, training, and real-time testing.

---

## Features

- **Custom Dataset Generation**: Capture and label your own images for "fake" and "real" classes.
- **Dataset Splitting**: Automatically splits the dataset into training, validation, and testing sets.
- **YOLO Training**: Train your YOLO model with custom settings like epochs.
- **Real-Time Testing**: Test the trained YOLO model using webcam data.

---

## Data Preparation

### 1. Generate Real and Fake Data
- Note that you need to create the following directories manually as the code will not create the directory ie dataset/split dataset/fake dataset/real all need to be created
- Run the `face_detector.py` script to collect and label data:
  - **Set the `classId`:**
    - Set `classId = 0` for **fake** data.
    - Set `classId = 1` for **real** data.
  - **Set the `outputFolderPath`:**
    - Use `dataset/fake` for **fake** data.
    - Use `dataset/real` for **real** data.
- Run the script:
  ```bash
  python face_detector.py

2. Split Dataset
	•	Modify the path: in the dataYaml variable to the absolute path of where your data.yaml will be saved.
	•	Run the data_split.py script to split the dataset into train, validation, and test sets and to generate a data.yaml file.

Example data.yaml:

path: /Users/samuel/ff/Liveness detector/dataset/split
train: train/images
val: val/images
test: test/images

nc: 2
names: ['fake', 'real']

Training
	•	Run the train.py script to train the YOLO model:
	•	Set the epochs parameter in the model.train() function to your desired value.

Testing
	•	Run the inference.py script to test the trained model on real-time webcam data:

python inference.py

