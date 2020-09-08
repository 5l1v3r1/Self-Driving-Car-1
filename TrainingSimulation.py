from utlis import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


path = "Data"
data = importDataInfo(path)
balanceData(data)

imagesPath, steering = loadData(path, data)
print(imagesPath[0], steering[0])