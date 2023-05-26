import pandas as pd
import matplotlib.pyplot as plt
import venn

# Wczytaj dane z pliku .csv
data = pd.read_csv("output - output (1).csv")

# Usuń wartości NaN z każdej kolumny
set1 = set(data["tcgaGD_data_protein"].dropna())
set2 = set(data["frame_tcgaGDC_data_GE_counts"].dropna())
set3 = set(data["frame_tcgaGD_data_Meth"].dropna())
set4 = set(data["frame_tcgaGDC_data_CNV_genelevel"].dropna())
set5 = set(data["frame_tcgaLGG_data_RNA_normalized"].dropna())
set6 = set(data["imaging"].dropna())

# Utwórz diagram Venn'a
labels = venn.get_labels([set1, set2, set3, set4, set5, set6], fill=['number', 'logic'])
fig, ax = venn.venn6(labels, names=["tcgaGD_data_protein", "frame_tcgaGDC_data_GE_counts", "frame_tcgaGD_data_Meth", "frame_tcgaGDC_data_CNV_genelevel", "frame_tcgaLGG_data_RNA_normalized", "imaging"], fontsize=8)

# Wyświetl diagram
plt.show()
