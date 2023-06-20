import os
import re
import cv2
import time
import pickle
import pydicom
import numpy as np
import pymrmr
import pandas as pd
from ReliefF import ReliefF
import scipy.stats as stats
from sklearn import metrics
import tensorflow as tf
import tensorflow_io as tfio
from keras import models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
from sklearn import model_selection, ensemble
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
from statsmodels.stats.multitest import multipletests
from skfeature.function.information_theoretical_based import FCBF
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import GroupKFold



'''
PYTANIA:
1. Czy zbiór jest za mały skoro nie liczy mi pewnych wartości metryk?
2. Dane kliniczne to tylko vital_status - co napisać w przetwarzaniu danych klinicznych itp?
3.
'''
