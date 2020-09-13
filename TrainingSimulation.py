from utlis import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path = "Data"
data = importDataInfo(path)
balanceData(data)

imagesPath, steering = loadData(path, data)
# print(imagesPath[0], steering[0])

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print("Total Training Images: ", len(xTrain))
print("Total Validation Images: ", len(xVal))









