# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:30:39 2023

@author: Lenovo
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
import time
from imgaug import augmenters as iaa
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

data = pd.read_csv("D:\GitHub\Self-Driving-Car\data\dataset_lake\driving_log.csv", names=["center_cam", "left_cam", "right_cam", "steering_angle", "throttle", "reverse", "speed"])

img_dir = "D:\GitHub\Self-Driving-Car\data\dataset_lake\IMG"

for i in range(len(data)):
    data["center_cam"][i] = data["center_cam"][i].split("\\")[-1]
    data["left_cam"][i] = data["left_cam"][i].split("\\")[-1]
    data["right_cam"][i] = data["right_cam"][i].split("\\")[-1]
    
def image_steer_data(img_dir, data):
    image_path = []
    steer = []
    for i in range(len(data)):
        idata = data.iloc[i]
        center, left, right = idata[0], idata[1], idata[2]
        image_path.append(os.path.join(img_dir,center))
        steer.append(idata[3])
        image_path.append(os.path.join(img_dir,left))
        steer.append(idata[3] + 0.15)
        image_path.append(os.path.join(img_dir,right))
        steer.append(idata[3] - 0.15)
    image_paths = np.array(image_path)
    steers = np.array(steer)
    return (image_paths, steers)
        
def preprocess_augment_data(img, angle):
    pp = 0
    image = cv2.imread(img)
    cv2.imshow("iamge",image)
    9
    #print("img: ", img)
    #print("iamge: ", image)
    image = cv2.resize(image, (200,66))
    image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if np.random.rand() < 0.25:
        image = cv2.flip(image,1)
        angle = -1 * float(angle)
        #angle = str(angle)
        pp = 1
        
    elif np.random.rand() >= 0.25 and np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        image = brightness.augment_image(image)
        pp = 2
        
    elif np.random.rand() >= 0.50 and np.random.rand() < 0.75:
        pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
        image = pan.augment_image(image)
        pp = 3
        
    else:
        zoom = iaa.Affine(scale=(1, 1.2))
        image = zoom.augment_image(image)
        pp = 4
    
    return image, angle

class DAVE2(nn.Module):
    def __init__(self):
        super(DAVE2, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = x.reshape(-1,1152)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0
    tt = 0
    for batch_idx, (img, steering) in enumerate(train_loader):
        steering = torch.from_numpy(np.asarray(steering))
        steering = torch.tensor(steering, dtype=torch.float32)
        #print("img: ", type(img))
        optimizer.zero_grad()
        img = img.permute(0,3,1,2)
        output = model(img)
        steering = torch.reshape(steering, (len(steering),1))
        loss = criterion(output, steering)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        tt += 1
        #print("#: ", tt)
    return train_loss / len(train_loader)


def validation(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for img, steering in test_loader:
            steering = torch.from_numpy(np.asarray(steering))
            steering = torch.tensor(steering, dtype=torch.float32)
            img = img.permute(0,3,1,2)
            output = model(img)
            steering = torch.reshape(steering, (len(steering),1))
            loss = criterion(output, steering)
            test_loss += loss.item()
    return test_loss / len(test_loader)

class DataSet(Dataset):
    def __init__(self, data_array):
        #print("init")
        self.data_array = data_array
        #self.imgg = data_array[:][0]
        #self.angg = data_array[:][1]
        #print(type(data_array))
        #print("dataa: ", data_array[][0])
        #print("imgg: ", self.imgg)
        #print("angg: ", self.angg)
        #print("lenimg: ", type(self.imgg))
        #print("lenang: ", type(self.angg))
                
    def __len__(self):
        #print("len")
        return len(self.data_array)
    
    def __getitem__(self, idx):
        #print("getitem")
        batch_x = self.data_array[idx][0]
        batch_y = self.data_array[idx][1]
        #print("b_x: ", batch_x)
        #print("b_y: ", batch_y)
        X, y = preprocess_augment_data(batch_x, float(batch_y))
        #print(X.shape, type(X))
        #print(f"X: {X}, y: {y}")
        return X, y

new_data = image_steer_data(img_dir, data)
new_data = np.array(new_data)
#datafile = DataSet(new_data)
lenData = len(new_data.T)
split = 0.2
num_val_samples = int(np.floor(split*lenData))
num_train_samples = lenData - num_val_samples
train_samples, val_samples = random_split(new_data.T, [num_train_samples, num_val_samples])

train_set = DataSet(train_samples)
train_Data = DataLoader(train_set, batch_size=64, shuffle=True)

val_set = DataSet(val_samples)
#print("trainset: ", train_set[0])


val_Data = DataLoader(val_set, batch_size=64, shuffle=True)
#print(next(iter(train_Data)))
#print("123")
model_path = None
smallest_loss = np.inf

model = DAVE2()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

max_epochs = 50

model_dir = "D:\GitHub\Self-Driving-Car\models"

for epoch in range(1, max_epochs+1):
    #print("456")
    train_loss = train(model, optimizer, criterion, train_Data)
    val_loss = validation(model, criterion, val_Data)
    #print("789")
    print(f"Epoch: {epoch}, train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")
    
    model_folder_name = f'epoch_{epoch:04d}_loss_{val_loss:.8f}'
    if not os.path.exists(os.path.join(model_dir, model_folder_name)):
        os.makedirs(os.path.join(model_dir, model_folder_name))
    torch.save(model.state_dict(), os.path.join(model_dir, model_folder_name, 'modeldrive.h5'))

    if model_path is None or val_loss < smallest_loss:
        smallest_loss = val_loss
        model_path = os.path.join(model_dir, model_folder_name, 'modeldrive.h5')

print(f'Model saved at {model_path}')






        
    



