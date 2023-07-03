from . import *
from memory_profiler import profile

class ModelBuilder:
    def __init__(self, filepath, X, y, n_splits=2, modelName=None, train_indices=None, test_indices=None, patient_ids=None):
        # Load data
        self.modelName = modelName
        self.skip = False
        if isinstance(filepath, np.ndarray):
            self.df = pd.DataFrame(filepath)
        else:
            self.df = pd.read_excel(filepath) if filepath is not None else None
        self.X = X
        self.y = y
        self.patient_ids = patient_ids
        self.patient_ids_for_proba = []
        self.probabilities = []

        #STATS
        self.n_splits = n_splits
        self.features = self.X.shape[1]
        self.featureNames = self.X.columns
        print(self.featureNames)
        self.scores = None

        if(self.modelName != None): print(f'=== BUILDING MODEL: {self.modelName} ===')
        if(self.features <= 1):
            print(f'Error -> Model unable to build')
            self.skip = True
            return

        # Convert Class column to 0 (Dead) and 1 (Alive)
        self.df['Class'] = self.df['Class'].map({'Dead': 0, 'Alive': 1})

        # StratifiedKFold
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        if(train_indices == None and test_indices == None):
            # Initialize placeholder for training and testing indices
            self.train_indices = []
            self.test_indices = []
        else:
            self.train_indices = train_indices
            self.test_indices = test_indices




    def pickle_save(self):
        with open(f'{self.modelName}.pkl', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def pickle_load(modelName):
        with open(f'{modelName}.pkl', 'rb') as file:
            return pickle.load(file)

    def cross_validate(self):
        #self.X = self.df.drop(['Class', 'ID'], axis=1)
        #self.y = self.df['Class']

        if(self.skip): return

        # Perform stratified cross-validation and store split indices
        for train_index, test_index in self.skf.split(self.X, self.y):
            self.train_indices.append(train_index)
            self.test_indices.append(test_index)

        print("Train indices:", self.train_indices)
        print("Test indices:", self.test_indices)

    @profile
    def train_and_evaluate(self, model_type='random_forest', metrics_list=['accuracy'], return_probabilities=False):

        if(self.skip): return

        start_time = time.time()

        model_dict = {
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'logistic_regression': LogisticRegression
        }

        # Wybierz model na podstawie model_type
        model_class = model_dict.get(model_type)
        if model_class is None:
            raise ValueError(f"Invalid model type: {model_type}. Valid options are: {list(model_dict.keys())}")

        # Utwórz instancję modelu
        model = model_class()

        # Reszta twojego kodu

        # Define dictionary of scoring metrics
        score_funcs = {
            'accuracy': metrics.accuracy_score,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'f1_score': metrics.f1_score,
            'roc_auc_score': metrics.roc_auc_score,
            'mcc': metrics.matthews_corrcoef
        }

        # Initialize dictionary for scores
        self.scores = {metric: [] for metric in metrics_list}
        self.probabilities = []
        self.decisions = []  # nowy atrybut przechowujący decyzje

        for train_index, test_index in zip(self.train_indices, self.test_indices):
            #print("Train indices:", train_index)
            #print("Test indices:", test_index)

            #self.X.reset_index(drop=True, inplace=True)

            print("Shape of X:", self.X.shape)
            print("Max train_index:", max(train_index))
            print("Max test_index:", max(test_index))

            print("Indeksy X:")
            print(self.X.index)

            print("train_index:")
            print(train_index)

            print("test_index:")
            print(test_index)

            # Split the data
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            print("X_train:")
            print(X_train)
            print("y_train:")
            print(y_train)
            print(np.unique(y_train))
            print("y_test:")
            print(y_test)
            print(np.unique(y_test))

            # Fit the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            end_time = time.time()
            print(f'Total training time: {abs(end_time - start_time)}s')

            # Evaluate the model and store scores
            for metric in metrics_list:
                score_func = score_funcs[metric]
                print(np.unique(y_test), np.unique(y_pred))
                #if(len(np.unique(y_test)) != 2 or len(np.unique(y_pred)) != 2):
                #    print(f'ZACZYNAM PONOWNIE...')
                #    self.train_and_evaluate(model_type=model_type, metrics_list=metrics_list, return_probabilities=return_probabilities)
                score = score_func(y_test, y_pred, zero_division=1) if metric in ["precision", "recall", "f1_score"] else score_func(y_test, y_pred)
                #print(f'{metric} score: {score}')  # Add this print statement
                self.scores[metric].append(score)

            if(return_probabilities):
                proba = [p[0] for p in model.predict_proba(X_test)]
                self.probabilities.append(proba)
                if self.patient_ids is not None:  # dodanie warunku
                    self.patient_ids_for_proba.append(self.patient_ids[test_index])
                    print(proba)
                    print(self.patient_ids_for_proba)
                self.decisions.append(y_test)

        # Print out the average scores over the folds
        for metric, values in self.scores.items():
            #print(f'{metric} values: {values}')  # Add this print statement

            if len(values) > 0:
                avg_score = sum(values) / len(values)
                print(f'Average {metric}: {avg_score}')
            else:
                print("Error: The list 'values' is empty. Cannot compute the average score.")

        else:
            return self.scores


class ImageModelBuilding:

    def __init__(self, X, y, n_splits=3, modelName='IMG'):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.skip = False
        self.kfold = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        self.probabilities = []

        self.modelName = modelName
        self.features = X.shape[1]  # Jeżeli chodzi o liczbę cech w obrazie, można to dostosować

    def pickle_save(self):
        with open(f'{self.modelName}.pkl', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def pickle_load(modelName):
        with open(f'{modelName}.pkl', 'rb') as file:
            return pickle.load(file)

    @profile
    def build_model(self, model_type=0):

        start_time = time.time()

        if model_type == 0:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(512, 512, 1)),
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
        elif model_type == 1:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(512, 512, 1)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        elif model_type == 2:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(512, 512, 1)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            raise ValueError("Invalid model_type. Must be 0, 1, or 2.")

        model.summary()
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                      loss=BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        end_time = time.time()
        print(f'Total training time: {abs(end_time - start_time)}s')

        return model


    def cross_validate(self, patient_ids, metrics_list=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'mcc']):
        group_kfold = GroupKFold(n_splits=self.n_splits)
        probabilities = []
        patient_ids_result = []

        self.X = np.transpose(self.X, (0, 2, 3, 1))

        score_funcs = {
            'accuracy': metrics.accuracy_score,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'f1_score': metrics.f1_score,
            'roc_auc_score': metrics.roc_auc_score,
            'mcc': metrics.matthews_corrcoef
        }

        # Initialize dictionary for scores
        self.scores = {metric: [] for metric in metrics_list}

        for train, test in group_kfold.split(self.X, self.y, groups=patient_ids):
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.y[train], self.y[test]

            model = self.build_model()
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)  # adjust epochs and batch_size to your needs
            proba = model.predict(X_test)
            y_pred = (proba > 0.5).astype(int)
            probabilities.extend((1 - proba).flatten())
            patient_ids_result.extend(patient_ids[test])

            # Evaluate the model and store scores
            for metric in metrics_list:
                score_func = score_funcs[metric]
                #if(len(np.unique(y_test)) != 2 or len(np.unique(y_pred)) != 2):
                #    print(f'ZACZYNAM PONOWNIE...')
                #    self.train_and_evaluate(model_type=model_type, metrics_list=metrics_list, return_probabilities=return_probabilities)
                score = score_func(y_test, y_pred, zero_division=1) if metric in ["precision", "recall", "f1_score"] else score_func(y_test, y_pred)
                #print(f'{metric} score: {score}')  # Add this print statement
                self.scores[metric].append(score)
                print(f'{metric}: {score}')

        results_df = pd.DataFrame({
            'id': patient_ids_result,
            'prob': probabilities
        })

        self.probabilities = results_df
        return results_df



class OmicsModelBuilding(ModelBuilder):
    def __init__(self, filepath, X, y, n_splits=3, modelName=None, train_indices=None, test_indices=None, patient_ids=None):
        super().__init__(filepath, X, y, n_splits, modelName=modelName, train_indices=train_indices, test_indices=test_indices, patient_ids=patient_ids)

    def train_and_evaluate(self, model_type='random_forest', metrics_list=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'mcc'], return_probabilities=False):
        super().train_and_evaluate(model_type=model_type, metrics_list=metrics_list, return_probabilities=return_probabilities)
