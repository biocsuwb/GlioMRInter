import os
import re
import cv2
import pickle
import pydicom
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
import tensorflow_io as tfio
from keras import models, layers
from keras.utils import to_categorical
from sklearn import model_selection, ensemble
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold


'''
PYTANIA:
1. Czy augmentacja powinna wystąpić wyłącznie raz przed kroswalidacją?
2. Jakie metryki, w jaki sposób?
'''
