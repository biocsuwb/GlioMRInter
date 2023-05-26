import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Wczytaj dane
data = pd.read_csv('df.CNV.merge.image.LGG.csv')

# Podziel dane na cechy (features) i etykiety (labels)
X = data.drop(['id', 'class'], axis=1)
y = data['class']

# Podziel dane na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utwórz pipeline, który najpierw przeskaluje dane, a potem zastosuje klasyfikator
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Zdefiniuj przestrzeń poszukiwań dla hiperparametrów
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
}

# Utwórz obiekt GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, cv=5, verbose=1, n_jobs=-1)

# Dopasuj model do danych treningowych
grid_search.fit(X_train, y_train)

# Wybierz najlepszy zestaw hiperparametrów
print("Najlepsze parametry: ", grid_search.best_params_)

# Przewiduj klasy dla danych testowych
y_pred = grid_search.predict(X_test)

# Oblicz i wydrukuj dokładność klasyfikatora
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
