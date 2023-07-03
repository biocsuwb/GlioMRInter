# Standardowe biblioteki
import os
import re
import time
import itertools
import pickle

# Biblioteki naukowe i obliczeniowe
import numpy as np
import pandas as pd
import scipy.stats as stats
import pydicom
import cv2
from sklearn import metrics, model_selection, ensemble
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, KFold
from statsmodels.stats.multitest import multipletests
from skfeature.function.information_theoretical_based import FCBF

# Biblioteki do przetwarzania obrazów i uczenia maszynowego
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import BinaryCrossentropy
from keras import models, layers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# Biblioteki do wyboru cech
import pymrmr
from ReliefF import ReliefF

# Biblioteki do wizualizacji
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3




'''
PYTANIA:
1. Czy zbiór jest za mały skoro nie liczy mi pewnych wartości metryk?
2. Dane kliniczne to tylko vital_status - co napisać w przetwarzaniu danych klinicznych itp?
3.


- ROBIĆ CZĘŚĆ WSPÓLNĄ NAJPIERW PRZED KROSWALIDACJĄ
- 2-krotna kroswalidacja
- wybór sieci konwolucyjnej
'''
