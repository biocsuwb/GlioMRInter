import os
import shutil
import pandas as pd

# ścieżki do folderów
input_folder = "E:/Magisterka/Dane"
output_folder = "D:/Magisterka/Dane"

# odczytanie pliku Excel z klasami
class_df = pd.read_excel("E:/Magisterka/IDs.xlsx", header=None, names=["ID", "Class"])
class_df["Class"].replace({"Alive": 1, "Dead": 0}, inplace=True)

# iteracja po podfolderach TCGA-GBM i TCGA-LGG
for cancer_type in os.listdir(input_folder):
    cancer_type_folder = os.path.join(input_folder, cancer_type)
    for patient_id in os.listdir(cancer_type_folder):
        # odczytanie klasy pacjenta
        class_row = class_df[class_df["ID"] == patient_id]
        if not class_row.empty:
            patient_class = class_row["Class"].iloc[0]
            # utworzenie ścieżki do katalogu wynikowego
            output_path = os.path.join(output_folder, str(patient_class), patient_id)
            os.makedirs(output_path, exist_ok=True)
            # iteracja po plikach .dcm w podfolderze pacjenta
            for root, dirs, files in os.walk(os.path.join(cancer_type_folder, patient_id)):
                for filename in files:
                    print(f'Weryfikuję plik {filename}...')
                    if filename.endswith(".dcm"):
                        # skopiowanie pliku .dcm do katalogu wynikowego
                        print(f'Kopiuję {filename} do {output_path}...')
                        shutil.copy(os.path.join(root, filename), output_path)
        else:
            print(f"Nie znaleziono klasy pacjenta {patient_id}")
