from . import *

class ModelBuilder:
    def __init__(self, filepath, X, y, n_splits=5, modelName=None, train_indices=None, test_indices=None):
        # Load data
        self.modelName = modelName
        self.skip = False
        self.df = pd.read_excel(filepath)
        self.X = X
        self.y = y
        self.probabilities = []

        #STATS
        self.n_splits = n_splits
        self.features = self.X.shape[1]
        self.scores = None

        if(self.modelName != None): print(f'=== BUILDING MODEL: {self.modelName} ===')
        if(self.features <= 0):
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

    def train_and_evaluate(self, model, metrics_list=['accuracy'], return_probabilities=False):

        if(self.skip): return

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

            self.X.reset_index(drop=True, inplace=True)

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

            #print("X_train:")
            #print(X_train)
            #print("y_train:")
            #print(y_train)



            # Fit the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model and store scores
            for metric in metrics_list:
                score_func = score_funcs[metric]
                score = score_func(y_test, y_pred, zero_division=1) if metric in ["precision", "recall", "f1_score"] else score_func(y_test, y_pred)
                #print(f'{metric} score: {score}')  # Add this print statement
                self.scores[metric].append(score)

            if(return_probabilities):
                proba = [p[0] for p in model.predict_proba(X_test)]
                self.probabilities.append(proba)
                print(proba)
                self.decisions.append(y_pred)

        # Print out the average scores over the folds
        for metric, values in self.scores.items():
            #print(f'{metric} values: {values}')  # Add this print statement
            avg_score = sum(values) / len(values)
            print(f'Average {metric}: {avg_score}')

        else:
            # Return the scores for all folds
            return self.scores


class ImageModelBuilding(ModelBuilder):
    def __init__(self, X, y, n_splits=5):
        super().__init__(X, y, n_splits)

    def build_model(self, number_of_classes):
        """
        Funkcja buduje model sieci konwolucyjnej
        """
        print(f'NOWY STATUS: Buduję model sieci...')
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=(512, 512, 1)),
            MaxPooling2D(2, 2),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def train_and_evaluate(self):
        model = self.build_model()
        super().train_and_evaluate(model, ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score'])



class OmicsModelBuilding(ModelBuilder):
    def __init__(self, filepath, X, y, n_splits=3, modelName=None, train_indices=None, test_indices=None):
        super().__init__(filepath, X, y, n_splits, modelName=modelName, train_indices=train_indices, test_indices=test_indices)

    def train_and_evaluate(self, return_probabilities=False):
        model = RandomForestClassifier()
        super().train_and_evaluate(model, ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'mcc'], return_probabilities=return_probabilities)
