import os
import re
import cv2
import pickle
import pydicom
import numpy as np
import pandas as pd
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

'''

import keras.backend as K
from sklearn.metrics import roc_auc_score, matthews_corrcoef

def precision(y_true, y_pred):
    """
    Oblicza precyzję klasyfikacji binarnej.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """
    Oblicza recall klasyfikacji binarnej.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    """
    Oblicza F1-score klasyfikacji binarnej.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_score = 2 * ((p * r) / (p + r + K.epsilon()))
    return f1_score

def mcc(y_true, y_pred):
    """
    Oblicza współczynnik korelacji Matthews'a (MCC).
    """
    mcc = matthews_corrcoef(K.eval(y_true), K.round(K.eval(y_pred)))
    return mcc

# Rejestracja funkcji jako metryk niestandardowych
from keras.utils import get_custom_objects
get_custom_objects().update({'precision': precision})
get_custom_objects().update({'recall': recall})
get_custom_objects().update({'f1_score': f1_score})
get_custom_objects().update({'auc': roc_auc_score})
get_custom_objects().update({'MCC': mcc})
