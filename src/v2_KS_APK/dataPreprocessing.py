from . import *

class DataPreprocessing:

    X = None
    y = None
    number_of_classes = None

    def __init__(self):
        pass

    def imagesPrep(self, data_path):
        self.ids = None#self.load_ids(ids_path)
        self.data_path = data_path
        self.X, self.y = self.read_dicom_images()

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

    def read_dicom_images(self):
        """
        Funkcja odczytuje pliki DICOM z podanego folderu i zwraca je jako tablicę numpy.
        """
        print(f'NOWY STATUS: Wczytuję zdjęcia w formacie DICOM...')
        images = []
        labels = []

        max_files_per_folder = 1000

        for folder_name in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder_name)
            if os.path.isdir(folder_path):
                num_files = 0
                for patient_id in os.listdir(folder_path):
                    patient_path = os.path.join(folder_path, patient_id)
                    if os.path.isdir(patient_path):
                        for file_name in os.listdir(patient_path):
                            file_path = os.path.join(patient_path, file_name)
                            if file_name.endswith(".dcm") and num_files < max_files_per_folder:
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
                                    num_files += 1
                                    print(f'[{patient_id}] Wczytano plik {file_name}. (Łącznie: {num_files} plików.)')
                                except Exception as e:
                                    print("Error reading file {}: {}".format(file_path, e))

        return np.array(images), np.array(labels)

    def read_dicom_imagesOLD(self):
        """
        Funkcja odczytuje pliki DICOM z podanego folderu i zwraca je jako tablicę numpy.
        """
        print(f'NOWY STATUS: Wczytuję zdjęcia w formacie DICOM...')
        images = []
        labels = []
        filesCounter = 0
        filesPerFolder = 5000

        startPercent = 0.6
        endPercent = 0.7
        quitFlag = False
        for root, dirs, files in os.walk(self.data_path):
            if (quitFlag): break

            print(f'Foldery: {dirs}')
            for dir in dirs:
                print(f'Wchodzę w folder "{dir}"...')
                if (quitFlag):
                    break
                for subdir,subdirs,subfiles in os.walk(os.path.join(root,dir)):
                    if (quitFlag):
                        break
                    if (len(subfiles) >= 5):
                        #print(f'Jest {len(subfiles)} zdjęć, a ja biorę tylko od {int(len(subfiles) * startPercent)} do {int(len(subfiles) * endPercent)}')
                        for subfile in subfiles[int(len(subfiles) * startPercent):int(len(subfiles) * endPercent)]:
                            if (filesCounter > filesPerFolder):
                                quitFlag = True
                                break
                            if subfile.endswith('.dcm'):
                                ds = pydicom.dcmread(os.path.join(subdir,subfile))
                                if not 'PixelData' in ds:
                                    continue

                                label = re.split("[\\\\/]", str(subdir))[4]
                                if self.get_id(label) not in ["Dead", "Alive"]:
                                    continue
                                else:
                                    labels.append(0 if self.get_id(label) == "Dead" else 1)

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

                                    print(f'[{label}] Wczytano: {filesCounter}/722347 plików ({round((filesCounter/722347) * 100, 2)}%)')
                                    filesCounter += 1
                            #break # <-- TYLKO JEDNO ZDJĘCIE

        for image in images:
            print(image.shape)

        self.number_of_classes = len(np.unique(labels))

        print(f'NOWY STATUS: Wczytywanie zakończono...')
        return np.array(images),np.array(labels)

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


'''
DO ZROBIENIA:
- Skrypt do przebudowania struktury danych wejściowych (wstępne przetwarzanie danych)

- Alive
-- TCA-02-314-2
--- 1.dcm
--- 2.dcm

'''
