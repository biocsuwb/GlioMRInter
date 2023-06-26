from . import *

class ImageDataPreprocessing:

    X = None
    y = None
    number_of_classes = None
    ids = None
    patient_ids = None

    def __init__(self):
        pass

    def imagesPrep(self, data_path, ids_path):
        self.ids = self.load_ids(ids_path)
        self.data_path = data_path
        self.X, self.y, self.patient_ids = self.read_dicom_images()
        #self.X, self.y = self.read_images()

    def load_ids(self, ids_path):
        print(f'NOWY STATUS: Wczytuję ID z pliku .xlsx...')
        return pd.read_excel(ids_path, header=None, names=['ID', 'VALUE'])

    def read_images(self):
        print(f'NOWY STATUS: Wczytuję zdjęcia...')

        images = []
        labels = []

        max_files_per_folder = 100

        for folder_name in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder_name)
            if os.path.isdir(folder_path):
                num_files = 0

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.endswith(".png") and num_files < max_files_per_folder:
                        try:
                            # Wczytanie pliku obrazu .png lub .jpg
                            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                            if image is not None:
                                # Skalowanie obrazu do wymiarów 512x512
                                image = cv2.resize(image, (512, 512))

                                # Dopasowanie obrazu do jednorodnego kształtu
                                image = image.reshape((1,) + image.shape)

                                images.append(image)
                                labels.append(int(folder_name))
                                num_files += 1
                                print(f'[{patient_id}] Wczytano plik {file_name}. (Łącznie: {num_files} plików.)')
                            else:
                                print(f"Nie udało się wczytać pliku {file_path}.")
                        except Exception as e:
                            print("Error reading file {}: {}".format(file_path, e))

        return np.array(images), np.array(labels)

    def read_dicom_images(self):
        """
        Funkcja odczytuje pliki DICOM z podanego folderu i zwraca je jako tablicę numpy.
        """
        print(f'NOWY STATUS: Wczytuję zdjęcia w formacie DICOM...')
        images = []
        labels = []
        patient_ids = []

        max_files_per_class = 99999
        max_files_per_patient = 1

        for folder_name in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder_name)
            if os.path.isdir(folder_path):
                num_files_per_class = 0
                for patient_id in os.listdir(folder_path):
                    patient_path = os.path.join(folder_path, patient_id)
                    if os.path.isdir(patient_path):
                        num_files_per_patient = 0
                        for file_name in os.listdir(patient_path):
                            if (num_files_per_patient < max_files_per_patient):
                                file_path = os.path.join(patient_path, file_name)
                                if file_name.endswith(".dcm") and num_files_per_class < max_files_per_class:
                                    try:
                                        ds = pydicom.dcmread(file_path)
                                        pixel_array = ds.pixel_array
                                        # skalowanie obrazu do wymiarów 512x512
                                        pixel_array = cv2.resize(pixel_array, (512, 512))
                                        # usuwanie kanałów koloru, pozostawienie tylko kanału zielonego
                                        if len(pixel_array.shape) > 2:
                                            pixel_array = pixel_array[:,:,1]
                                        # Dopasowanie obrazu do jednorodnego kształtu
                                        pixel_array = pixel_array.reshape((1,) + pixel_array.shape)

                                        image = pixel_array
                                        #image = cv2.resize(image, (512, 512))
                                        images.append(image)
                                        labels.append(int(folder_name))
                                        patient_ids.append(patient_id)
                                        num_files_per_class += 1
                                        num_files_per_patient += 1
                                        print(f'[{patient_id}] Wczytano plik {file_name}. (Łącznie: {num_files_per_class} plików.)')
                                    except Exception as e:
                                        print("Error reading file {}: {}".format(file_path, e))

        return np.array(images), np.array(labels), np.array(patient_ids)

    def augment_data(self, x_train, y_train):
        """
        Funkcja przekształca dane wejściowe (zdjęcia) i wyjściowe (etykiety)
        """
        print(f'NOWY STATUS: Augmentuję dane...')
        data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )

        x_train = np.expand_dims(x_train, axis=-1)
        data_gen.fit(x_train)

        return data_gen.flow(x_train, y_train, batch_size=32)

class OmicDataPreprocessing:

    def __init__(self, path=None, df=None, sep=';', decimal=','):
        self.path = path
        self.df = df
        self.ID = None
        self.X = None
        self.y = None
        self.columns = None
        self.omic_data = None
        self.sep = sep
        self.decimal = decimal

    def load_data(self):
        self.omic_data = pd.read_csv(self.path, sep=sep, decimal=decimal) if(self.path != None) else self.df

    def Xy_data(self):
        if self.df is None or self.df.empty:
            print(len(self.omic_data))
            if 'id' in self.omic_data.columns:
                self.X = self.omic_data.drop(columns=["class", "id"])
                self.ID = self.omic_data["id"]  # Store ID as attribute
                print(self.ID)
            else:
                self.X = self.omic_data.drop(columns=["class"])
            self.y = self.omic_data["class"]
            self.columns = self.X.columns
        else:
            if 'id' in self.df.columns:
                self.X = self.df.drop(columns=["class", "id"])
                self.ID = self.df["id"]  # Store ID as attribute
                print(self.ID)
            else:
                self.X = self.df.drop(columns=["class"])
            self.y = self.df["class"]
            self.columns = self.X.columns

    def normalize_data(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(self.X, columns=self.columns)

    def remove_redundant_features(self, correlation_threshold=0.75):
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        self.X = self.X.drop(columns=to_drop)  # Drop redundant features

    def feature_selection(self, method=None, n_features=100, correlation_threshold=0.5):
        if method == 'mrmr':
            old = self.X.shape[1]
            selected_features = pymrmr.mRMR(self.X, 'MIQ', n_features)
            self.X = self.X[selected_features]
            self.remove_redundant_features(correlation_threshold)
            print(f'{old} -> [MRMR] -> {self.X.shape[1]}')

        elif method == 'relief':
            old = self.X.shape[1]
            fs = ReliefF(n_neighbors=10, n_features_to_keep=n_features)
            X_numpy = self.X.values
            transformed_X = fs.fit_transform(X_numpy, self.y)
            feature_scores = fs.feature_scores
            sorted_indices = np.argsort(feature_scores)[::-1]
            selected_feature_names = self.X.columns[sorted_indices[:n_features]]
            self.X = pd.DataFrame(transformed_X, columns=selected_feature_names)
            self.remove_redundant_features(correlation_threshold)
            print(f'{old} -> [ReliefF] -> {self.X.shape[1]}')

        elif method == 'utest':
            old = self.X.shape[1]
            class_0 = self.X[self.y == 0]
            class_1 = self.X[self.y == 1]
            p_values = {}
            for column in self.X.columns:
                u_statistic, p_value = stats.mannwhitneyu(class_0[column], class_1[column])
                p_values[column] = p_value
            _, p_value_adjusted, _, _ = multipletests(list(p_values.values()), method='fdr_bh')
            selected_features = [column for column, adjusted_p_value in zip(p_values.keys(), p_value_adjusted) if adjusted_p_value < 0.05]
            self.X = self.X[selected_features]
            self.remove_redundant_features(correlation_threshold)
            print(f'{old} -> [U-Test] -> {self.X.shape[1]}')

        elif method == 'fcbf':
            old = self.X.shape[1]

            # Obliczanie współczynnika korelacji FCBF
            idx = FCBF.fcbf(self.X.values, self.y.values)

            # Wybór cech
            selected_features = self.X.columns[idx[0]]

            # Aktualizacja X do tylko wybranych cech
            self.X = self.X[selected_features]

            # Usunięcie nadmiarowych cech
            self.remove_redundant_features(correlation_threshold)

            print(f'{old} -> [FCBF] -> {self.X.shape[1]}')
