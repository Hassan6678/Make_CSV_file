import DataSet
import csv
import os
import glob
import cv2 as cv
import features
import time
from tqdm import tqdm

Train_fileName = 'CNN_train_value.csv'
Test_fileName = 'CNN_test_value.csv'

def writeCSV_train(value):
    if os.path.exists(Train_fileName):
        with open(Train_fileName, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter = ',' ,quoting=csv.QUOTE_ALL)
            writer.writerow(value,)
    else:
        print("NO file exist")
def writeCSV_test(value):
    if os.path.exists(Test_fileName):
        with open(Test_fileName, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter = ',' ,quoting=csv.QUOTE_ALL)
            writer.writerow(value,)
    else:
        print("NO file exist")

def header():
    head = []
    img_row = 28; img_col = 28
    for val in range((img_row * img_col)+1):
        if val == 0:
            head.append("lebel")
        else:
            head.append("pixel"+str(val))
    return head

header_value = header()

def create_file(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)  # if old file exist first delete that file then create New File
        with open(fileName, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file,delimiter = ',' ,quoting=csv.QUOTE_ALL)
            writer.writerow(header_value)
    else:  # Otherwise we create new file
        with open(fileName, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file,delimiter = ',' ,quoting=csv.QUOTE_ALL)
            writer.writerow(header_value)

dir_list = DataSet.data_dir()

Name = []
for name in dir_list:
    f = name.split('\\')
    Name.append(f[-1])

create_file(Train_fileName)
create_file(Test_fileName)

target = 0
for folder in range(len(dir_list)):
    data_path = os.path.join(dir_list[folder], '*.png')
    files = glob.glob(data_path)
    path = []
    if len(files) == 0:
        continue
    train = []
    test = []
    img_count = 1
    for f1 in files:
        img = cv.imread(f1,0)
        if img_count % 3 == 0:
            test.append(img)
        else:
            train.append(img)

        img_count += 1
    target += 1

    sequence = 1
    print(Name[target - 1], "Train progressing...")
    for image in train:
        de_noise = features.remove_noise(image)
        binary = features.Binarize_image(de_noise)
        b_w = features.Convert_B_W(binary)
        cnn = features.CNN(b_w)
        cnn.insert(0, int(target - 1))
        writeCSV_train(cnn)
        sequence += 1

    for i in tqdm(range(sequence-1)):
        time.sleep(0.01)

    sequence = 1
    print(Name[target - 1], "Test progressing...")
    for image in test:
        de_noise = features.remove_noise(image)
        binary = features.Binarize_image(de_noise)
        b_w = features.Convert_B_W(binary)
        cnn = features.CNN(b_w)
        cnn.insert(0, int(target - 1))
        writeCSV_test(cnn)
        sequence += 1

    for i in tqdm(range(sequence - 1)):
        time.sleep(0.01)

print("\nWork Done")