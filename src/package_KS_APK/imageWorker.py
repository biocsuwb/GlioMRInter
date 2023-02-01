import os
import re
import cv2
import pydicom
import numpy as np
import pandas as pd
from keras import models, layers
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold

class ImageWorker:

    def __init__(self, data_path):
        self.ids = None
        self.data_path = data_path
        self.number_of_classes = -1

    def load_data(self, file_path):
        print(f'NOWY STATUS: Wczytuję ID z pliku .xlsx...')
        self.ids = pd.read_excel(file_path, header=None, names=['ID', 'VALUE'])
        print(self.ids)

    def get_value(self, label):
        if self.ids is None:
            print("Dataframe 'ids' nie został wczytany.")
            return None

        try:
            return self.ids[self.ids['ID'] == label]['VALUE'].values[0]
        except IndexError:
            return None

    def read_dicom_images(self):
        """
        Funkcja odczytuje pliki DICOM z podanego folderu i zwraca je jako tablicę numpy.
        """
        print(f'NOWY STATUS: Wczytuję zdjęcia w formacie DICOM...')
        images = []
        labels = []
        filesCounter = 0
        filesPerFolder = 2000

        startPercent = 0.6
        endPercent = 0.7
        quitFlag = False
        for root, dirs, files in os.walk(self.data_path):
            if (quitFlag): break

            print(f'Foldery: {dirs}')
            for dir in dirs:
                print(f'Wchodzę w folder "{dir}"...')
                if (quitFlag):
                    break
                for subdir,subdirs,subfiles in os.walk(os.path.join(root,dir)):
                    if (quitFlag):
                        break
                    if (len(subfiles) >= 5):
                        #print(f'Jest {len(subfiles)} zdjęć, a ja biorę tylko od {int(len(subfiles) * startPercent)} do {int(len(subfiles) * endPercent)}')
                        for subfile in subfiles[int(len(subfiles) * startPercent):int(len(subfiles) * endPercent)]:
                            if (filesCounter > filesPerFolder):
                                quitFlag = True
                                break
                            if subfile.endswith('.dcm'):
                                ds = pydicom.dcmread(os.path.join(subdir,subfile))

                                label = re.split("[\\\\/]", str(subdir))[4]
                                if self.get_value(label) not in ["Dead", "Alive"]:
                                    continue
                                else:
                                    labels.append(self.get_value(label))

                                    image = ds.pixel_array
                                    image = cv2.resize(image, (512, 512))
                                    images.append(image)

                                    print(f'[{label}] Wczytano: {filesCounter}/722347 plików ({round((filesCounter/722347) * 100, 2)}%)')
                                    filesCounter += 1

        print(f'NOWY STATUS: Wczytywanie zakończono...')
        return np.array(images),np.array(labels)

    def preprocessing(self):
        print(f'NOWY STATUS: Rozpoczynam preprocessing...')
        X, y = self.read_dicom_images()
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = X_train.reshape(X_train.shape[0], 512, 512, 1)
        X_test = X_test.reshape(X_test.shape[0], 512, 512, 1)
        self.number_of_classes = len(np.unique(y_train))
        return X_train, X_test, y_train, y_test

    def build_model(self):
        """
        Funkcja buduje model sieci konwolucyjnej
        """
        print(f'NOWY STATUS: Buduję model sieci...')
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.number_of_classes, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def augment_data(self, x_train, y_train):
        """
        Funkcja przekształca dane wejściowe (zdjęcia) i wyjściowe (etykiety)
        """
        print(f'NOWY STATUS: Augmentuję dane...')
        data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )

        data_gen.fit(x_train)

        return data_gen.flow(x_train, y_train, batch_size=32)

    def cross_validate(self, x, y, n_splits=5):
        """
        Funkcja zastosowuje kroswalidację stratyfikowaną i augmentację danych, po
        czym trenuje i waliduje model.
        """
        print(f'NOWY STATUS: Przeprowadzam kroswalidację...')
        skf = StratifiedKFold(n_splits=n_splits)

        for train_index, val_index in skf.split(x, y):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            augmented_data = self.augment_data(x_train, y_train)
            model = self.build_model()
            model.fit(augmented_data, epochs=10)

            val_loss, val_acc = model.evaluate(x_val, y_val)
            print("Validation Loss: ", val_loss)
            print("Validation Acc: ", val_acc)

if __name__ == '__main__':
    print(f'NOWY STATUS: Rozpoczynam działanie...')
    data_path = 'D:/Magisterka/Dane'
    image_worker = ImageWorker(data_path)
    image_worker.load_data('D:/Magisterka/IDs.xlsx')
    X_train, X_test, y_train, y_test = image_worker.preprocessing()
    image_worker.cross_validate(X_train, y_train)
