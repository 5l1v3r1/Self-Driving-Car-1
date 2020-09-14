print("Setting UP!")
import os
import socketio
import eventlet
from flask import Flask
from PIL import Image
from io import BytesIO
import base64
import cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from utlis import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense

path = "Data"
data = importDataInfo(path)
balanceData(data)

imagesPath, steering = loadData(path, data)
# print(imagesPath[0], steering[0])

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print("Total Training Images: ", len(xTrain))
print("Total Validation Images: ", len(xVal))

model = createModel()
model.summary()

# Training here
history = model.fit(batchGen(xTrain, yTrain, 10, 1), steps_per_epoch=300, epochs=10,
                    validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

model.save("model.h5")
print("Model Saved!")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Training", "Validation"])
plt.ylim([0,1])
plt.title("Loss")
plt.xlabel("Epoch")
plt.show()
