from v2_KS_APK import dataPreprocessing as dp
from v2_KS_APK import modelBuilding as mb
from v2_KS_APK import dataVisualization as dv
import time
import pandas as pd
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
data = dp.ImageDataPreprocessing()
modelBuilder = mb.ImageModelBuilding()

data.imagesPrep('D:/Magisterka/Dane_LGG', "E:/Magisterka/AllIDs.xlsx")
#data.imagesPrep('D:/Magisterka/Dane')

model = modelBuilder.build_model(data.number_of_classes)
modelBuilder.cross_validation(data.X, data.y, model, data)

'''

method = "utest"
features = 50
id_path = "E:/Magisterka/AllIDs.xlsx"
probabilities = True

def run(method, features, id_path="E:/Magisterka/AllIDs.xlsx", probabilities=True):
# START
    print(f'{method}, {features}')

    data_CNA = dp.OmicDataPreprocessing(path='correctData/df.CNV.merge.image.LGG.csv')
    data_METH = dp.OmicDataPreprocessing(path='correctData/df.METH.merge.image.LGG.csv')
    data_RNA = dp.OmicDataPreprocessing(path='correctData/df.RNA.merge.image.LGG.csv')
    data_RPPA = dp.OmicDataPreprocessing(path='correctData/df.RPPA.merge.image.LGG.csv')

    data_CNA.load_data()
    data_METH.load_data()
    data_RNA.load_data()
    data_RPPA.load_data()

    data_CNA.omic_data = data_CNA.omic_data.drop_duplicates(subset='id', keep='first')
    data_METH.omic_data = data_METH.omic_data.drop_duplicates(subset='id', keep='first')
    data_RNA.omic_data = data_RNA.omic_data.drop_duplicates(subset='id', keep='first')
    data_RPPA.omic_data = data_RPPA.omic_data.drop_duplicates(subset='id', keep='first')

    # Inicjujemy common_ids jako pusty zestaw
    common_ids = set(data_CNA.omic_data['id'])

    for data in [data_METH, data_RNA, data_RPPA]:
        common_ids = common_ids.intersection(set(data.omic_data['id']))

    print(common_ids)

    print("CNA duplicates:", data_CNA.omic_data['id'].duplicated().any())
    print("METH duplicates:", data_METH.omic_data['id'].duplicated().any())
    print("RNA duplicates:", data_RNA.omic_data['id'].duplicated().any())
    print("RPPA duplicates:", data_RPPA.omic_data['id'].duplicated().any())

    print(len(data_CNA.omic_data), "<-")
    data_CNA.omic_data = data_CNA.omic_data[data_CNA.omic_data['id'].isin(common_ids)]
    print(len(data_CNA.omic_data), "<-")
    data_CNA.omic_data = data_CNA.omic_data.reset_index(drop=True)
    print(len(data_CNA.omic_data), "<- =====")

    print(len(data_METH.omic_data), "<-")
    data_METH.omic_data = data_METH.omic_data[data_METH.omic_data['id'].isin(common_ids)]
    print(len(data_METH.omic_data), "<-")
    data_METH.omic_data = data_METH.omic_data.reset_index(drop=True)
    print(len(data_METH.omic_data), "<- =====")

    print(len(data_RNA.omic_data), "<-")
    data_RNA.omic_data = data_RNA.omic_data[data_RNA.omic_data['id'].isin(common_ids)]
    print(len(data_RNA.omic_data), "<-")
    data_RNA.omic_data = data_RNA.omic_data.reset_index(drop=True)
    print(len(data_RNA.omic_data), "<- =====")

    print(len(data_RPPA.omic_data), "<-")
    data_RPPA.omic_data = data_RPPA.omic_data[data_RPPA.omic_data['id'].isin(common_ids)]
    print(len(data_RPPA.omic_data), "<-")
    data_RPPA.omic_data = data_RPPA.omic_data.reset_index(drop=True)
    print(len(data_RPPA.omic_data), "<- =====")




    data_CNA.Xy_data()
    data_CNA.normalize_data()
    data_CNA.feature_selection(method=method, n_features=features)

    data_METH.Xy_data()
    data_METH.normalize_data()
    data_METH.feature_selection(method=method, n_features=features)

    data_RNA.Xy_data()
    data_RNA.normalize_data()
    data_RNA.feature_selection(method=method, n_features=features)

    data_RPPA.Xy_data()
    data_RPPA.normalize_data()
    data_RPPA.feature_selection(method=method, n_features=features)

    #MODEL

    print(data_RPPA.ID)

    trainer_RPPA_timeStart = time.time()
    trainer_RPPA = mb.OmicsModelBuilding(id_path, data_RPPA.X, data_RPPA.y, modelName="RPPA", patient_ids=data_RPPA.ID)
    trainer_RPPA.cross_validate()
    trainer_RPPA.train_and_evaluate(model_type='random_forest', return_probabilities=probabilities)
    trainer_RPPA.pickle_save()
    trainer_RPPA_timeStop = trainer_RPPA_timeStop = time.time()
    trainer_RPPA_time = trainer_RPPA_timeStop - trainer_RPPA_timeStart

    trainer_METH_timeStart = time.time()
    trainer_METH = mb.OmicsModelBuilding(id_path, data_METH.X, data_METH.y, modelName="METH", train_indices=trainer_RPPA.train_indices, test_indices=trainer_RPPA.test_indices, patient_ids=data_RPPA.ID)
    trainer_METH.train_and_evaluate(model_type='random_forest', return_probabilities=probabilities)
    trainer_METH.pickle_save()
    trainer_METH_timeStop = time.time()
    trainer_METH_time = trainer_METH_timeStop - trainer_METH_timeStart

    trainer_RNA_timeStart = time.time()
    trainer_RNA = mb.OmicsModelBuilding(id_path, data_RNA.X, data_RNA.y, modelName="RNA", train_indices=trainer_RPPA.train_indices, test_indices=trainer_RPPA.test_indices, patient_ids=data_RPPA.ID)
    trainer_RNA.train_and_evaluate(model_type='random_forest', return_probabilities=probabilities)
    trainer_RNA.pickle_save()
    trainer_RNA_timeStop = time.time()
    trainer_RNA_time = trainer_RNA_timeStop - trainer_RNA_timeStart

    trainer_CNA_timeStart = time.time()
    trainer_CNA = mb.OmicsModelBuilding(id_path, data_CNA.X, data_CNA.y, modelName="CNA", train_indices=trainer_RPPA.train_indices, test_indices=trainer_RPPA.test_indices, patient_ids=data_RPPA.ID)
    trainer_CNA.train_and_evaluate(model_type='random_forest', return_probabilities=probabilities)
    trainer_CNA.pickle_save()
    trainer_CNA_timeStop = time.time()
    trainer_CNA_time = trainer_CNA_timeStop - trainer_CNA_timeStart


    # OBRAZY

    data = dp.ImageDataPreprocessing()
    data.imagesPrep('D:/Magisterka/Dane_LGG', "E:/Magisterka/AllIDs.xlsx")

    X, y = data.X, data.y

    trainer_IMG_timeStart = time.time()
    trainer_IMG = mb.ImageModelBuilding(X, y)
    model = trainer_IMG.build_model()
    images_prob = trainer_IMG.cross_validate(data.patient_ids)
    trainer_IMG.pickle_save()
    trainer_IMG_timeStop = trainer_IMG_timeStop = time.time()
    trainer_IMG_time = trainer_IMG_timeStop - trainer_IMG_timeStart

    print(images_prob)
    '''
    trainer_CNA = mb.ModelBuilder.pickle_load("CNA")
    trainer_METH = mb.ModelBuilder.pickle_load("METH")
    trainer_RNA = mb.ModelBuilder.pickle_load("RNA")
    trainer_RPPA = mb.ModelBuilder.pickle_load("RPPA")
    trainer_IMG = mb.ImageModelBuilding.pickle_load("IMG")

    plot = dv.DataVisualizer([trainer_RPPA], method, features)
    plot.visualize_models()

    plot = dv.DataVisualizer([trainer_METH], method, features)
    plot.visualize_models()

    plot = dv.DataVisualizer([trainer_RNA], method, features)
    plot.visualize_models()

    plot = dv.DataVisualizer([trainer_CNA], method, features)
    plot.visualize_models()

    plot = dv.DataVisualizer([trainer_IMG], method, features)
    plot.visualize_models()
    '''

    probabilities_df = pd.DataFrame()

    print(trainer_RPPA.decisions)
    print(trainer_RPPA.patient_ids)

    probabilities_df['class'] = np.concatenate(trainer_RPPA.decisions).flatten()
    probabilities_df['id'] = trainer_RPPA.patient_ids
    if(trainer_CNA.probabilities): probabilities_df['CNA_prob'] = np.concatenate(trainer_CNA.probabilities).flatten()
    if(trainer_METH.probabilities): probabilities_df['METH_prob'] = np.concatenate(trainer_METH.probabilities).flatten()
    if(trainer_RNA.probabilities): probabilities_df['RNA_prob'] = np.concatenate(trainer_RNA.probabilities).flatten()
    if(trainer_RPPA.probabilities): probabilities_df['RPPA_prob'] = np.concatenate(trainer_RPPA.probabilities).flatten()


    probabilities_df.reset_index(drop=True, inplace=True)
    print(probabilities_df)
    trainer_IMG.probabilities.rename(columns={'prob': 'IMG_prob'}, inplace=True)
    all_df = probabilities_df.merge(trainer_IMG.probabilities, on='id', how='left')
    clinical_df = pd.read_csv('correctData/LGG.clinical.id.vitalstatus.csv', sep=';', decimal=',')
    clinical_df.rename(columns={'bcr_patient_barcode': 'id'}, inplace=True)
    subset_df = clinical_df[['vital_status', 'id']]
    all_df = all_df.merge(subset_df, on='id', how='left')

    data_ALL_timeStart = time.time()
    data_ALL = dp.OmicDataPreprocessing(df=all_df)
    data_ALL.load_data()
    data_ALL.Xy_data()
    data_ALL_timeStop = time.time()
    data_ALL_time = data_ALL_timeStop - data_ALL_timeStart

    trainer_ALL_timeStart = time.time()
    trainer_ALL = mb.OmicsModelBuilding(id_path, data_ALL.X, data_ALL.y, modelName="ALL")
    trainer_ALL.cross_validate()
    trainer_ALL.train_and_evaluate(model_type='random_forest', return_probabilities=probabilities)
    trainer_ALL_timeStop = time.time()
    trainer_ALL_time = trainer_ALL_timeStop - trainer_ALL_timeStart

    #plot = dv.DataVisualizer([trainer_RNA, trainer_METH, trainer_CNA, trainer_RPPA, trainer_IMG, trainer_ALL], method, features)
    #plot.visualize_models()
    #plot.boxplot('accuracy')
    #plot.venn_plot()
    #plot.feature_dependency_plot()

    return trainer_ALL


#t10 = run("utest", 10)
#t20 = run("utest", 20)
t50 = run("utest", 50)
#t100 = run("utest", 100)
#t150 = run("utest", 150)

# Lista modeli
models = [t50]

# Lista wartości cech
features = [50]

# Lista metryk
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'mcc', 'mean_squared_error']

for metric in metrics:
    data = []
    for i, model in enumerate(models):
        avg_score = sum(model.scores[metric])/len(model.scores[metric])
        data.append({'Liczba cech': features[i], 'Score': avg_score, 'Metric': metric})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x='Liczba cech', y='Score', data=df, marker='o')
    plt.title(f'Zmiana wartości metryki {metric} w zależności od liczby cech')
    plt.show()
