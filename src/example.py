from v2_KS_APK import dataPreprocessing as dp
from v2_KS_APK import modelBuilding as mb
import time
import numpy as np


# Utwórz obiekt do przetwarzania danych
data = dp.ImageDataPreprocessing()

# Przygotuj dane
data.imagesPrep('D:/Magisterka/Dane_LGG', "E:/Magisterka/AllIDs.xlsx")

# Pobierz dane X i y
X, y = data.X, data.y

modelBuilder = mb.ImageModelBuilding(X, y)
model = modelBuilder.build_model()  # zakładając, że liczba unikalnych wartości y to liczba klas
modelBuilder.cross_validate(data.patient_ids)

'''

# Przygotowanie danych
preparingTimeStart = time.time()
preprocessor = dp.OmicDataPreprocessing(path='data/RNA.csv')
preprocessor.load_data()
preprocessor.normalize_data()
preprocessor.feature_selection(method="relief", n_features=100)
preparingTime = time.time() - preparingTimeStart

# Trenowanie i ocena modelu
modelBuildingTimeStart = time.time()
trainer = mb.OmicsModelBuilding("E:/Magisterka/AllIDs.xlsx", preprocessor.X, preprocessor.y)
trainer.train_and_evaluate()
modelBuildingTime = time.time() - modelBuildingTimeStart

print(f'Czas przygotowywania danych: {preparingTime} \nCzas budowania modelu: {modelBuildingTime}')
'''

'''
Kroswalidacja na poczatku VVV
Poprawic omiczne, zmienic nazwe VVV
Sprawdzić na danych z artykułu
Dopiero puścić na glejaku

Dodać liczenie czasu
Moduł do wizualizacji wyników (wykresy skuteczności w zależności od liczby cech, w zależności od selekcji)

===
Zapisywanie do pickle w milestoneach
Zapisywać do CSV wyniki

====

MDFS1D <-- dodać zamiast MRMR!!!
fast correlation based filter FCBF <-- !!!
model zintegrowany!!!
'''
