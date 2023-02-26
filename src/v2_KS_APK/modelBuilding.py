from . import *

class ModelBuilding:

    def __init__(self):
        #self.clf_rf = ensemble.RandomForestClassifier()

    def build_model(self, number_of_classes):
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
        model.add(layers.Dense(number_of_classes, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def cross_validation(self, X, y, model, data, n_splits=3, test_size=0.3):

        sss = model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        splits = sss.split(X, y)

        for i, (train_id, test_id) in enumerate(sss.split(X, y)):
            print(f"Fold {i}:")
            print(f"  Train: index={train_id}")
            print(f"  Test:  index={test_id}")

            X_train = X[train_id]
            X_test = X[test_id]
            y_train = y[train_id]
            y_test = y[test_id]

            augmented_data = data.augment_data(X_train, y_train)
            model.fit(augmented_data, epochs=10)

            #print(f'X_train:\n{X_train}\n\ny_train:\n{y_train}')

            #self.clf_rf.fit(X_train, y_train)
            #self.clf_rf.score(X_test, y_test)

    def model_write(self):

        with open('./classifierRF.pickle', 'wb') as fileFR:
            pickle.dump(self.clf_rf, fileFR)

    def model_read(self):

        with open('./classifierRF.pickle', 'rb') as fileFR:
            loaded_clf = pickle.load(fileFR)
