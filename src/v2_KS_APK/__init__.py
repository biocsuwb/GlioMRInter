import os
import re
import cv2
import pickle
import pydicom
import numpy as np
import pymrmr
import pandas as pd
from ReliefF import ReliefF
import scipy.stats as stats
import sklearn.metrics as metrics
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


'''
PYTANIA:
1. Czy dawać do wyboru jeden selektor cech czy możliwość użycia wszystkich po kolei?
2. Ile n_features_to_keep w ReliefF?

Wektor prawdopodobieństwa

transcriptomprofile
copynumbervariation

u-test

Transcriptome Profiling

GBM
tcgaGDC_data_CNV_genelevel.Rds
tcgaGD_data_RNA_normalized.Rds
tcgaGD_data_protein.Rds
tcgaGD_data_Meth.Rds
tcgaGDC_data_GE_counts.Rds

Venn Plot!

Z każdych danych tabela, póxniej wybierzemy przypadki

1. Sprawdzić ID które się pokrywają i zrobić tabele z danymi
2. Pierwszą kolumnę zrobić zmienną decyzyjną
3. Przetestować obrazy

Zmienna decyzyjna jak coś w tcgaGD_data_clin.Rds

'''
