# IntegraCIO: python package for clinical, multi-omics and image glioma data integration and analysis
## Description
IntegraCIO is a system for integrating large clinical, omics and imaging data sets to support the diagnosis or prognosis of patients with central nervous system tumors using advanced supervised learning methods, i.e. deep neural networks and machine learning.

**IntegraCIO is a Python package that allows the user to:**
* filter the most informative biomarkers from molecular data generated from high-throughput experiments;
* remove redundant and correlated features (biomarkes) from the obtained feature subsets;
* build and evaluate individual predictive model (binary task) with multiple omics data by using the random forest or support vector classifier (SVC) algorithms and machine learning validation techniques;
* build and evaluate individual predictive model (binary task) with image data by using the convolutional neural network (CNN) and machine learning validation techniques;
* build and evaluate the ensemble predictive model (binary task) with clinical and syntetic variables (omics and image) by using the random forest or support vector classifier (SVC) algorithms and machine learning validation techniques;
* perform data augmentation for image data;
* offer various stability measures for assessing the stability of a given method on a given data set in comparison with other methods.
* establish the selected parameters for predictive models, such as: ;
* save and visualize the data and model results in the form of plots and tables;

![Fig.1](https://github.com/biocsuwb/Images/blob/main/Scheme1G.png?raw=true)
Fig.1 The IntegraCIO scheme.

# Description of used data sets

To demonstrate the functionality of IntegraCIO the different type of molecular data and image data from the Cancer Genome Atlas Low Grade Glioma (TCGA-LGG) project were used.
Raw data sets can be download from The Cancer Genome Atlas database ([TCGA](https://www.cancer.gov/tcga)) and The Cancer Imaging Program database ([TCIA]([The Cancer Imaging Program](https://www.cancerimagingarchive.net/)))
The following type of data were used:
- clinical data (CD);
- gene expression profiles (GE) obtained with Illumina Human HT-12 v3 microarray;
- copy-number alterations data (CNA) obtained with Affymetrix SNP 6.0;
- mRNA levels of gene expression (RNA-seq V2 RSEM normalized expression values);
- DNA methylation profiles (METH) generated from Illumina HM450K array (beta-values for genes);
- protein expression profiling with reverse-phase protein arrays (RPPA); 
- MRI image data.

The preprocessing of molecular data involved standard steps, namely, the log2 transformation of data was performed and the features with zero and near-zero (1%) variance across patients were removed. The data from disparate sources were consolided and merged by the ID patients. For testing purposes, the number of molecular markers was limited to 2000 DEGs ranked by the highest difference in the gene expression level between tumor and normal tissues ([exampleData_TCGA_LUAD_2000.csv](https://github.com/biocsuwb/EnsembleFS-package/tree/main/data)). 


## Install the development version from GitHub:
To install this package, clone the repository and install with pip:
```r
git clone https://github.com/IntegraCIO/IntegraCIO
cd IntegraCIO
pip install .

or:

pip install IntegraCIO==1.0
```
## Import module
```r
from IntegraCIO import dataPreprocessing as dp
from IntegraCIO import modelBuilding as mb
from IntegraCIO import dataVisualization as dv
import time
import pandas as pd
import numpy as np
```

### Load omics data
```r
download.file("https://raw.githubusercontent.com/biocsuwb/EnsembleFS-package/main/data/correctData/df.METH.merge.image.LGG.csv", 
              destfile = "correctData/df.METH.merge.image.LGG.csv", method = "curl")

data_CNA = dp.OmicDataPreprocessing(path='correctData/df.CNV.merge.image.LGG.csv')
data_METH = dp.OmicDataPreprocessing(path='correctData/df.METH.merge.image.LGG.csv')
data_RNA = dp.OmicDataPreprocessing(path='correctData/df.RNA.merge.image.LGG.csv')
data_RPPA = dp.OmicDataPreprocessing(path='correctData/df.RPPA.merge.image.LGG.csv')

data_CNA.load_data()
data_METH.load_data()
data_RNA.load_data()
data_RPPA.load_data()
```

## Example 1 - Construct the predictive model with molecular data by using the hybrid feature selection approach

### Prepare omic data for machine learning
#### Remove duplicates from data sets (optional).
```r
data_CNA.omic_data = data_CNA.omic_data.drop_duplicates(subset='id', keep='first')
data_METH.omic_data = data_METH.omic_data.drop_duplicates(subset='id', keep='first')
data_RNA.omic_data = data_RNA.omic_data.drop_duplicates(subset='id', keep='first')
data_RPPA.omic_data = data_RPPA.omic_data.drop_duplicates(subset='id', keep='first')

```
#### Integrate data - merge data sets with ID samples (optional)
```r
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
####  Split your original dataset into predictors (X) and descriptor (y) and normalize X data (optional)

```r
data_CNA.Xy_data()
data_CNA.normalize_data()

data_METH.Xy_data()
data_METH.normalize_data()

data_RNA.Xy_data()
data_RNA.normalize_data()

data_RPPA.Xy_data()
data_RPPA.normalize_data()
```

### Perform feature selection (optional)
#### Input parameters of feature_selection() method from dataPreprocessing module
Syntax:
.feature_selection(method = None, n_features = 100, correlation_treshold = 0.75)
Input parameter values:
- feature selection methods: ***method = "utest", "mdfs", "fcbf", "relief", "mrmr"***;
- number of selected Ntop relevant features: ***n_features = 100***;
- Spearman's rank correlation coefficient <0, 1>: ***correlation_treshold = 0.75***;
```
#### Set up the configuration parameters for selected feature selection method
```r
- probabilities = True
```
#### Option 1 -  use a one feature filter to select Ntop features from molecular data sets
```r
data_CNA.feature_selection(method="mdfs", n_features=100)
data_METH.feature_selection(method="mdfs", n_features=100)
data_RNA.feature_selection(method="mdfs", n_features=100)
data_RPPA.feature_selection(method="mdfs", n_features=100)
```
#### Option 2 -  use a different feature filters to select Ntop relevant features from molecular data sets
```r
data_CNA.feature_selection(method="mdfs", n_features=100)
data_METH.feature_selection(method="fcbf", n_features=100)
data_RNA.feature_selection(method="relief", n_features=100)
data_RPPA.feature_selection(method="mrmr", n_features=100)
```
### Build and evaluate the predictive model with omic data
#### Input parameters of OmicsModelBuilding class from modelBuilding module
Syntax:
OmicsModelBuilding(id_path, data_RPPA.X, data_RPPA.y, modelName="RPPA", patient_ids=data_RPPA.ID)
Input parameters:
- path to .xlsx or .csv format file with id of all samples: ***id_path***;
- model name: ***modelName***;
- - ***train_indices***: indices of previous build model;
- ***test_indices***: indices of previous evaluated model;
- id samples: ***patient_ids***;
#### Input parameters of train_and_evaluate() method from modelBuilding module
Syntax: 
.train_and_evaluate(model_type='random_forest', return_probabilities=True)
Input parameters:
- binary classifier: ***model_type = "random_forest", "svc", "logistic_reg"***;
- ***return_probabilities***: model return the values of prediction vector as probabilities for return_probabilities=True and as logit values for return_probabilities=False;

```r
id_path = "E:/models/AllIDs.xlsx"

trainer_RPPA = mb.OmicsModelBuilding(id_path = id_path, train_indices = data_RPPA.X, test_indices = data_RPPA.y, modelName="RPPA", patient_ids=data_RPPA.ID)
trainer_RPPA.cross_validate()
trainer_RPPA.train_and_evaluate(model_type='random_forest', return_probabilities=probabilities)

# save model result (optional) 
trainer_RPPA.pickle_save()

trainer_METH = mb.OmicsModelBuilding(id_path, data_METH.X, data_METH.y, modelName="METH", train_indices=trainer_RPPA.train_indices, test_indices=trainer_RPPA.test_indices, patient_ids=data_RPPA.ID)
trainer_METH.train_and_evaluate(model_type='random_forest', return_probabilities=probabilities)

```
### Build and evaluate the predictive model with image data
#### Input parameters of ImageDataPreprocessing class from modelBuilding module
Syntax: modelBuilding.OmicsModelBuilding(id_path, data_RPPA.X, data_RPPA.y, modelName="RPPA", patient_ids=data_RPPA.ID)
Input parameters:
- path to .xlsx or .csv format file with id of all samples: ***id_path***;
- model name: ***modelName***;
- id samples: ***patient_ids***;

###  Prepare image data for CNN learning 
```r
data = dp.ImageDataPreprocessing()
data.imagesPrep('D:/Magisterka/Dane_LGG', "E:/Magisterka/AllIDs.xlsx")
X, y = data.X, data.y

### Build and evaluate the predictive model
trainer_IMG = mb.ImageModelBuilding(X, y)
model = trainer_IMG.build_model() 
images_prob = trainer_IMG.cross_validate(data.patient_ids)
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

## Example 2 - To compare the performance of prediction models or/and select the best performing predictive model the individual models can be build and evaluate
