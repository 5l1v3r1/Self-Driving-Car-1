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