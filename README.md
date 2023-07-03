# Nazwa Twojego Projektu

## Opis
Krótki opis Twojego projektu. Opis powinien być zwięzły, ale jednocześnie wystarczająco szczegółowy, aby ktoś, kto go przeczyta, zrozumiał, co Twój projekt robi.

## Instalacja
Aby zainstalować ten pakiet, sklonuj repozytorium i zainstaluj za pomocą pip:

git clone https://github.com/GlioMRInter/GlioMRInter
cd GlioMRInter
pip install .

## Użycie  

Tu znajdują się przykładowe użycia Twojego projektu. Właściwe przykłady pomagają innym zrozumieć, jak mogą korzystać z Twojego kodu. Przykład:

Najwygodniej rozpocząć pracę z pakietem od zdefiniowania pewnych wartości, które łatwiej będzie modyfikować w przyszłości i które znajdą zastosowanie jako przyszłe argumenty funkcji wywoływanych w skrypcie. Poniżej przedstawiono te wartości:

```
method = "utest"
features = 50
id_path = "E:/Magisterka/AllIDs.xlsx"
probabilities = True
```

Następnie wczytujemy i ładujemy dane omiczne, a także usuwamy duplikaty z tych zbiorów.

```
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

```

W kolejnym kroku modyfikujemy wszystkie dane omiczne, aby ze wszystkich zbiorów uchwycić część wspólną. Jest to dobra praktyka, która zapobiega ewentualnym problemom z identyfikatorami i rekordami w zbiorach danych.

```
common_ids = set(data_CNA.omic_data['id'])

for data in [data_METH, data_RNA, data_RPPA]:
    common_ids = common_ids.intersection(set(data.omic_data['id']))

data_CNA.omic_data = data_CNA.omic_data[data_CNA.omic_data['id'].isin(common_ids)]
data_CNA.omic_data = data_CNA.omic_data.reset_index(drop=True)

data_METH.omic_data = data_METH.omic_data[data_METH.omic_data['id'].isin(common_ids)]
data_METH.omic_data = data_METH.omic_data.reset_index(drop=True)

data_RNA.omic_data = data_RNA.omic_data[data_RNA.omic_data['id'].isin(common_ids)]
data_RNA.omic_data = data_RNA.omic_data.reset_index(drop=True)

data_RPPA.omic_data = data_RPPA.omic_data[data_RPPA.omic_data['id'].isin(common_ids)]
data_RPPA.omic_data = data_RPPA.omic_data.reset_index(drop=True)

```

Ostatnim krokiem odnoszącym się do wstępnego przetwarzania danych jest dla każdego zbioru podzielenie danych na cechy i zmienne decyzyjne, a także przeprowadzenie selekcji cech. Opcjonalnie można w tym momencie również przeprowadzić normalizację danych wejściowych.

```
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

```

Budowa modeli odbywa się z wykorzystaniem zewnętrznej kroswalidacji. Mamy pewność, że zewnętrzna kroswalidacja mimo zastosowania różnych typów danych (przy odpowiadających sobie indeksach) zadziała jak powinna. Dla każdego zbioru zostaje zbudowany, wytrenowany i zwalidowany model, który następnie przechowuje wyprodukowane zmienne syntetyczne. Poniżej, celem uniknięcia długich listingów kodu, przedstawiono dwa pierwsze modele. W pierwszym z nich przeprowadzana jest funkcja cross_validate(), natomiast w każdym kolejnym już zamiast niej przekazywane są pola z numerami indeksów dokładnie z tego modelu, który tą funkcję wywołał.

```
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

```

Kod odnoszący się do wczytywania, trenowania i walidacji danych obrazowych wygląda podobnie. Na koniec również uzyskujemy wektor prawdopodobieństw (dane syntetyczne), który następnie stanowić będzie jedną z cech danych tabelarycznych modelu zintegrowanego.

```
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

```

Fragment kodu przedstawia integrację danych klinicznych, omicznych i obrazowych w celu utworzenia modelu zintegrowanego. Wykorzystuje do tego utworzone wcześniej modele oraz dodatkowe dane kliniczne.

```
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

```

## Licencja

Ten projekt jest licencjonowany na podstawie licencji MIT. Pełny tekst licencji znajduje się w pliku LICENSE.

## Autorzy

Kacper Stasiełuk; kacperstasieluk@gmail.com

## Wkład 

Jeśli chcesz wnieść wkład do tego projektu, możesz to zrobić, tworząc nowy "pull request". Prosimy o skontaktowanie się z nami przed rozpoczęciem pracy, aby upewnić się, że Twój wkład jest zgodny z kierunkiem, w którym podąża projekt.