import os
import random
import shutil
from itertools import islice

outputFolderPath = "dataset/split"
inputFolderPathFake = "dataset/fake"
inputFolderPathReal = "dataset/real"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# In case the script is run several times, delete the existing versions of your train/test data first
try:
    shutil.rmtree(outputFolderPath)
except OSError as e:
    os.mkdir(outputFolderPath)

os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

fake_images = os.listdir(inputFolderPathFake)
real_images = os.listdir(inputFolderPathReal)
image_names = []
for img in fake_images:
    if img.endswith('.jpg'): 
        image_names.append((img.split('.')[0], 'fake'))
for img in real_images:
    if img.endswith('.jpg'): 
        image_names.append((img.split('.')[0], 'real'))

random.shuffle(image_names)

data_length = len(image_names)
train_length = int(data_length * splitRatio['train'])
validation_length = int(data_length * splitRatio['val'])
test_length = int(data_length * splitRatio['test'])

if data_length != train_length + test_length + validation_length:
    remaining = data_length - (train_length + test_length + validation_length)
    train_length += remaining

lengthToSplit = [train_length, validation_length, test_length]
Input = iter(image_names)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{data_length} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName, label in out:
        inputFolder = inputFolderPathFake if label == 'fake' else inputFolderPathReal
        
        image_path = f'{inputFolder}/{fileName}.jpg'
        label_path = f'{inputFolder}/{fileName}.txt'
        
        if os.path.exists(image_path) and os.path.exists(label_path):
            shutil.copy(image_path, f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
            shutil.copy(label_path, f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')
        else:
            print(f"File not found: {image_path} or {label_path}. Skipping...")


dataYaml = f'path: ..//Users/samuel/ff/Liveness detector/dataset/split\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

with open(f"{outputFolderPath}/data.yaml", 'a') as f:
    f.write(dataYaml)

