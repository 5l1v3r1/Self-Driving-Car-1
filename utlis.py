import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def getName(filePath):
    return filePath.split("\\")[-1]


def importDataInfo(path):
    columns = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names=columns)
    data["Center"] = data["Center"].apply(getName)
    # print(data["Center"])
    print("Total images imported: ", data.shape[0])
    return data


def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data["Steering"], nBins)

    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDataList = []

        for i in range(len(data["Steering"])):
            if data["Steering"][i] >= bins[j] and data["Steering"][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print("Removed images:", len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print("Remaining Images:", len(data))

    if display:
        hist, _ = np.histogram(data["Steering"], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()