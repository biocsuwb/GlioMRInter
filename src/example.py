from v2_KS_APK import dataPreprocessing as dp
from v2_KS_APK import modelBuilding as mb

data = dp.DataPreprocessing()
modelBuilder = mb.ModelBuilding()

data.imagesPrep('D:/Magisterka/Dane', 'D:/Magisterka/IDs.xlsx')

model = modelBuilder.build_model(data.number_of_classes)
modelBuilder.cross_validation(data.X, data.y, model, data)
