from v2_KS_APK import dataPreprocessing as dp
from v2_KS_APK import modelBuilding as mb
import time

'''
data = dp.ImageDataPreprocessing()
modelBuilder = mb.ImageModelBuilding()

data.imagesPrep('D:/Magisterka/Dane_LGG', "E:/Magisterka/AllIDs.xlsx")
#data.imagesPrep('D:/Magisterka/Dane')

model = modelBuilder.build_model(data.number_of_classes)
modelBuilder.cross_validation(data.X, data.y, model, data)

'''

# Przygotowanie danych
preparingTimeStart = time.time()
preprocessor = dp.OmicDataPreprocessing(path='data_train_RNA_integ_5typeDat.csv')
preprocessor.load_data()
preprocessor.normalize_data()
preprocessor.feature_selection(method="mrmr", n_features=100)
preparingTime = time.time() - preparingTimeStart

# Trenowanie i ocena modelu
modelBuildingTimeStart = time.time()
trainer = mb.OmicsModelBuilding("E:/Magisterka/AllIDs.xlsx", preprocessor.X, preprocessor.y)
trainer.train_and_evaluate()
modelBuildingTime = time.time() - modelBuildingTimeStart

print(f'Czas przygotowywania danych: {preparingTime} \nCzas budowania modelu: {modelBuildingTime}')


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
'''
