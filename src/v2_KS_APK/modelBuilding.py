from . import *

class ModelBuilding:

    def __init__(self):
        #self.clf_rf = ensemble.RandomForestClassifier()
        pass

    def build_model(self, number_of_classes):
        """
        Funkcja buduje model sieci konwolucyjnej
        """
        print(f'NOWY STATUS: Buduję model sieci...')
        model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                      input_shape=(512, 512, 1)),
          tf.keras.layers.MaxPooling2D(2, 2),
          tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.summary()

        model.compile(optimizer=RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                     metrics=['accuracy'])

                     #, 'precision', 'recall', 'f1_score', 'auc', 'MCC'
                     #tf.keras.metrics.Accuracy(), tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()

        '''
        - Accuracy: procent poprawnych predykcji względem całkowitej liczby próbek.
        - Binary accuracy: procent poprawnych predykcji w przypadku zadania binarnej klasyfikacji.
        - Precision: stosunek liczby poprawnie sklasyfikowanych pozytywnych przykładów do wszystkich pozytywnych predykcji.
        - Recall: stosunek liczby poprawnie sklasyfikowanych pozytywnych przykładów do wszystkich faktycznie pozytywnych przykładów w zbiorze.
        - F1 score: średnia harmoniczna precyzji i recall.
        - AUC: pole pod krzywą ROC (Receiver Operating Characteristic), której osią x jest false positive rate (FPR), a osią y jest true positive rate (TPR). AUC daje nam informację o tym, jak dobrze model radzi sobie w rozróżnianiu klas.
        '''


        return model

    def cross_validation(self, X, y, model, data, n_splits=3, test_size=0.3):

        #print(f'X: {X}, Y:{y}')
        newX = np.concatenate(X, axis=0)

        sss = model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        splits = sss.split(X, y)

        for i, (train_id, test_id) in enumerate(sss.split(newX, y)):
            print(f"Fold {i}:")
            print(f"  Train: index={train_id}")
            print(f"  Test:  index={test_id}")

            X_train = newX[train_id]
            X_test = newX[test_id]
            y_train = y[train_id]
            y_test = y[test_id]

            augmented_data = data.augment_data(X_train, y_train)
            print(augmented_data[0])
            print(f'NOWY STATUS: Trenuję dane...')
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
