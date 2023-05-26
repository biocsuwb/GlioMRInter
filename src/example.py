from v2_KS_APK import dataPreprocessing as dp
from v2_KS_APK import modelBuilding as mb

'''
data = dp.ImageDataPreprocessing()
modelBuilder = mb.ImageModelBuilding()

data.imagesPrep('D:/Magisterka/Dane_LGG', "E:/Magisterka/AllIDs.xlsx")
#data.imagesPrep('D:/Magisterka/Dane')

model = modelBuilder.build_model(data.number_of_classes)
modelBuilder.cross_validation(data.X, data.y, model, data)

'''
# Przykład użycia

# Przygotowanie danych
preprocessor = dp.OmicDataPreprocessing(path='df.CNV.merge.image.LGG.csv')
preprocessor.load_data()
preprocessor.normalize_data()
preprocessor.feature_selection(method="mrmr")

# Trenowanie i ocena modelu
trainer = mb.OmicsModelBuilding(preprocessor.X, preprocessor.y)
trainer.train_and_evaluate()
