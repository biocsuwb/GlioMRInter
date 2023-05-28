from . import *

class ImageDataPreprocessing:

    X = None
    y = None
    number_of_classes = None

    def __init__(self):
        pass

    def imagesPrep(self, data_path, ids_path):
        self.load_ids(ids_path)
        self.data_path = data_path
        self.X, self.y = self.read_dicom_images()
        #self.X, self.y = self.read_images()

    def load_ids(self, ids_path):
        print(f'NOWY STATUS: Wczytuję ID z pliku .xlsx...')
        return pd.read_excel(ids_path, header=None, names=['ID', 'VALUE'])

    def get_id(self, label):
        if self.ids is None:
            print("Dataframe 'ids' nie został wczytany.")
            return None

        try:
            return self.ids[self.ids['ID'] == label]['VALUE'].values[0]
        except IndexError:
            return None

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

        max_files_per_class = 500
        max_files_per_patient = 10

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
                                        num_files_per_class += 1
                                        num_files_per_patient += 1
                                        print(f'[{patient_id}] Wczytano plik {file_name}. (Łącznie: {num_files_per_class} plików.)')
                                    except Exception as e:
                                        print("Error reading file {}: {}".format(file_path, e))

        return np.array(images), np.array(labels)

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

    def __init__(self, path):
        self.path = path
        self.X = None
        self.y = None
        self.columns = None

    def load_data(self):
        omic_data = pd.read_csv(self.path, sep=';', decimal=',')
        self.X = omic_data.drop(columns=["class", "id"])
        self.y = omic_data["class"]
        self.columns = self.X.columns

    def normalize_data(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(self.X, columns=self.columns)

    def feature_selection(self, method=None, n_features=100):
        if method == 'mrmr':
            old = self.X.shape[1]
            selected_features = pymrmr.mRMR(self.X, 'MIQ', n_features) #zrobić opcjonalnie
            self.X = self.X[selected_features]
            print(f'{old} -> [MRMR] -> {self.X.shape[1]}')
        elif method == 'relief':
            old = self.X.shape[1]
            fs = ReliefF(n_neighbors=10, n_features_to_keep=n_features)
            self.X = fs.fit_transform(self.X.values, self.y)
            self.X = pd.DataFrame(self.X)
            new = self.X.shape[1]
            print(f'{old} -> [ReliefF] -> {new}')
        elif method == 'utest':
            old = self.X.shape[1]
            class_0 = self.X[self.y == 0]
            class_1 = self.X[self.y == 1]
            p_values = {}
            for column in self.X.columns:
                u_statistic, p_value = stats.mannwhitneyu(class_0[column], class_1[column])
                p_values[column] = p_value
            selected_features = [column for column, p_value in p_values.items() if p_value < 0.05]
            #poprawka na wielotesty, Holm, FDR <--
            # usuwanie redundatnych cech, parametr corr
            self.X = self.X[selected_features]
            print(f'{old} -> [U-Test] -> {self.X.shape[1]}')
