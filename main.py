import DataSet
import csv
import os
import glob
import cv2 as cv
import features
import time
from tqdm import tqdm

def writeCSV(area,ratio,solidity,m1,m2,m3,m4,m5,m6,m7,target):
    if os.path.exists('features.csv'):
        with open('features.csv', mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([area, ratio, solidity, m1, m2, m3, m4, m5, m6, m7, target])
    else:
        print("NO file exist")

if os.path.exists('features.csv'):
    os.remove('features.csv')  # if old file exist first delete that file then create New File
    with open('features.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Area', 'Aspect Ratio', 'Solidity', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'Target'])
else:  # Otherwise we create new file
    with open('features.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Area', 'Aspect Ratio', 'Solidity', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'Target'])

dir_list = DataSet.data_dir()

Name = []
for name in dir_list:
    f = name.split('\\')
    Name.append(f[-1])

target = 0
for folder in range(len(dir_list)):
    data_path = os.path.join(dir_list[folder], '*.png')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = cv.imread(f1, 0)
        data.append(img)
    target += 1

    sequence = 1
    print(Name[target - 1], "progressing...")
    for image in data:
        de_noise = features.remove_noise(image)

        b_img = features.Binarize_image(de_noise)

        b_to_w = features.Convert_B_W(b_img)

        m = features.Moments(b_to_w)

        ratio = round(features.Aspect_Ratio(b_to_w), 6)

        img = cv.cvtColor(b_to_w, cv.COLOR_GRAY2BGR)

        n_img = img.copy()

        _, solidity, convex_image = features.convex_hull(n_img)
        solidity = round(solidity, 2)
        area = features.cal_Area(b_to_w)

        writeCSV(area, ratio, solidity, float(m[0]), float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5]),float(m[6]), target-1)

        sequence += 1

    for i in tqdm(range(sequence-1)):
        time.sleep(0.01)


print("\nWork Done")
