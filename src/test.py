from v2_KS_APK import dataPreprocessing as dp
from v2_KS_APK import modelBuilding as mb
from v2_KS_APK import dataVisualization as dv
import time
import pandas as pd
import numpy as np

'''
data = dp.ImageDataPreprocessing()
modelBuilder = mb.ImageModelBuilding()

data.imagesPrep('D:/Magisterka/Dane_LGG', "E:/Magisterka/AllIDs.xlsx")
#data.imagesPrep('D:/Magisterka/Dane')

model = modelBuilder.build_model(data.number_of_classes)
modelBuilder.cross_validation(data.X, data.y, model, data)

'''

# START
method = "relief"
features = 100
id_path = "E:/Magisterka/AllIDs.xlsx"
probabilities = True

data_CNA_timeStart = time.time()
#data_CNA = dp.OmicDataPreprocessing(path='correctData/df.CNV.merge.image.LGG.csv')
data_CNA = dp.OmicDataPreprocessing(path='data/CNA.csv')
data_CNA.load_data()
data_CNA.normalize_data()
data_CNA.feature_selection(method=method, n_features=features)
data_CNA_timeStop = time.time()
data_CNA_time = data_CNA_timeStop - data_CNA_timeStart

data_METH_timeStart = time.time()
#data_METH = dp.OmicDataPreprocessing(path='correctData/df.METH.merge.image.LGG.csv')
data_METH = dp.OmicDataPreprocessing(path='data/METH.csv')
data_METH.load_data()
data_METH.normalize_data()
data_METH.feature_selection(method=method, n_features=features)
data_METH_timeStop = time.time()
data_METH_time = data_METH_timeStop - data_METH_timeStart

data_RNA_timeStart = time.time()
#data_RNA = dp.OmicDataPreprocessing(path='correctData/df.RNA.merge.image.LGG.csv')
data_RNA = dp.OmicDataPreprocessing(path='data/RNA.csv')
data_RNA.load_data()
data_RNA.normalize_data()
data_RNA.feature_selection(method=method, n_features=features)
data_RNA_timeStop = time.time()
data_RNA_time = data_RNA_timeStop - data_RNA_timeStart

data_RPPA_timeStart = time.time()
#data_RPPA = dp.OmicDataPreprocessing(path='correctData/df.RPPA.merge.image.LGG.csv')
data_RPPA = dp.OmicDataPreprocessing(path='data/RPPA.csv')
data_RPPA.load_data()
data_RPPA.normalize_data()
data_RPPA.feature_selection(method=method, n_features=features)
data_RPPA_timeStop = time.time()
data_RPPA_time = data_RPPA_timeStop - data_RPPA_timeStart

#print(f'\nTIMES:\n- CNA: {data_CNA_time}s\n- METH: {data_METH_time}s\n- RNA: {data_RNA_time}s\n- RPPA: {data_RPPA_time}s')

#MODEL

trainer_RNA_timeStart = time.time()
trainer_RNA = mb.OmicsModelBuilding(id_path, data_RNA.X, data_RNA.y, modelName="RNA")
trainer_RNA.cross_validate()
trainer_RNA.train_and_evaluate(return_probabilities=probabilities)
trainer_RNA.pickle_save()
trainer_RNA_timeStop = time.time()
trainer_RNA_time = trainer_RNA_timeStop - trainer_RNA_timeStart

trainer_CNA_timeStart = time.time()
trainer_CNA = mb.OmicsModelBuilding(id_path, data_CNA.X, data_CNA.y, modelName="CNA", train_indices=trainer_RNA.train_indices, test_indices=trainer_RNA.test_indices)
trainer_CNA.train_and_evaluate(return_probabilities=probabilities)
trainer_CNA.pickle_save()
trainer_CNA_timeStop = time.time()
trainer_CNA_time = trainer_CNA_timeStop - trainer_CNA_timeStart

trainer_METH_timeStart = time.time()
trainer_METH = mb.OmicsModelBuilding(id_path, data_METH.X, data_METH.y, modelName="METH", train_indices=trainer_RNA.train_indices, test_indices=trainer_RNA.test_indices)
trainer_METH.train_and_evaluate(return_probabilities=probabilities)
trainer_METH.pickle_save()
trainer_METH_timeStop = time.time()
trainer_METH_time = trainer_METH_timeStop - trainer_METH_timeStart


trainer_RPPA_timeStart = time.time()
trainer_RPPA = mb.OmicsModelBuilding(id_path, data_RPPA.X, data_RPPA.y, modelName="RPPA", train_indices=trainer_RNA.train_indices, test_indices=trainer_RNA.test_indices)
trainer_RPPA.train_and_evaluate(return_probabilities=probabilities)
trainer_RPPA.pickle_save()
trainer_RPPA_timeStop = trainer_RPPA_timeStop = time.time()
trainer_RPPA_time = trainer_RPPA_timeStop - trainer_RPPA_timeStart

# utworzenie nowego DataFrame
probabilities_df = pd.DataFrame()

# dodajemy kolumny prawdopodobieństw dla każdego z modeli
probabilities_df['class'] = np.concatenate(trainer_RNA.decisions).flatten()
if(trainer_CNA.probabilities): probabilities_df['CNA_prob'] = np.concatenate(trainer_CNA.probabilities).flatten()
if(trainer_METH.probabilities): probabilities_df['METH_prob'] = np.concatenate(trainer_METH.probabilities).flatten()
if(trainer_RNA.probabilities): probabilities_df['RNA_prob'] = np.concatenate(trainer_RNA.probabilities).flatten()
#if(trainer_RPPA.probabilities): probabilities_df['RPPA_prob'] = np.concatenate(trainer_RPPA.probabilities).flatten()

# dodajemy identyfikatory pacjentów jako indeks DataFrame
probabilities_df.reset_index(drop=True, inplace=True)#probabilities_df.set_index(data_CNA.df.index, inplace=True)  # zakładamy, że data_CNA.df.index zawiera identyfikatory pacjentów

print(probabilities_df)

data_ALL_timeStart = time.time()
data_ALL = dp.OmicDataPreprocessing(df=probabilities_df)
data_ALL.load_data()
#data_ALL.normalize_data()
#data_ALL.feature_selection(method="mrmr", n_features=features)
data_ALL_timeStop = time.time()
data_ALL_time = data_ALL_timeStop - data_ALL_timeStart

trainer_ALL_timeStart = time.time()
trainer_ALL = mb.OmicsModelBuilding(id_path, data_ALL.X, data_ALL.y, modelName="ALL")
trainer_ALL.cross_validate()
trainer_ALL.train_and_evaluate(return_probabilities=probabilities)
trainer_ALL_timeStop = time.time()
trainer_ALL_time = trainer_ALL_timeStop - trainer_ALL_timeStart

plot = dv.DataVisualizer([trainer_RNA, trainer_METH, trainer_CNA, trainer_RPPA], method, features)
plot.visualize_models()
plot.boxplot('accuracy')
plot.venn_plot()
plot.feature_dependency_plot()
