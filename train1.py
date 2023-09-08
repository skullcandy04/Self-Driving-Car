import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data  = pd.read_csv("D:\GitHub\Self-Driving-Car\data\dataset_lake\driving_log.csv", names=["center_cam", "left_cam", "right_cam", "steering_angle", "throttle", "reverse", "speed"])
img_dir = "D:\GitHub\Self-Driving-Car\data\dataset_lake\IMG"

for i in range(len(data)):
    data["center_cam"][i] = img_dir + "\\" + data["center_cam"][i].split("\\")[-1]
    data["left_cam"][i] = img_dir + "\\" + data["left_cam"][i].split("\\")[-1]
    data["right_cam"][i] = img_dir + "\\" + data["right_cam"][i].split("\\")[-1]


pic0_dir = data["center_cam"][0]

print(pic0_dir)

img = cv2.imread('D:\GitHub\Self-Driving-Car\data\dataset_lake\IMG\center_2022_04_10_12_24_41_840.jpg')
cv2.imshow("img", img)