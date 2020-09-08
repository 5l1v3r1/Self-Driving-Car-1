from utlis import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


path = "beta_simulator_windows/data"
data = importDataInfo(path)
balanceData(data)

loadData(path, data)