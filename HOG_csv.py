import DataSet
import csv
import os
import glob
import cv2 as cv
import features
import time
from tqdm import tqdm

fileName = 'HOG_value.csv'

def writeCSV(hog_value):
    if os.path.exists(fileName):
        with open(fileName, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter = ',' ,quoting=csv.QUOTE_ALL)
            writer.writerow(hog_value,)
    else:
        print("NO file exist")

if os.path.exists(fileName):
    os.remove(fileName)  # if old file exist first delete that file then create New File
    with open(fileName, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
else:  # Otherwise we create new file
    with open(fileName, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

dir_list = DataSet.data_dir()

Name = []
for name in dir_list:
    f = name.split('\\')
    Name.append(f[-1])

target = 0
for folder in range(len(dir_list)):
    data_path = os.path.join(dir_list[folder], '*.png')
    files = glob.glob(data_path)
    path = []
    if len(files) == 0:
        continue
    data = []
    for f1 in files:
        img = cv.imread(f1)
        data.append(img)
    target += 1

    sequence = 1
    print(Name[target - 1], "progressing...")
    for image in data:
        de_noise = features.remove_noise(image)
        hog = features.HOG(de_noise)
        hog.insert(0, int(target-1))
        #cnn = features.CNN(de_noise)
        #cnn.insert(0, int(target - 1))
        writeCSV(hog)
        sequence += 1

    for i in tqdm(range(sequence-1)):
        time.sleep(0.01)


print("\nWork Done")