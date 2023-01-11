import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# Wszystkie rysunki zostaną przeskalowane o wartość 1/255
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
  training_dir,
  target_size=(300, 300),
  class_mode='binary'
)

training_dir = 'horse-or-human/training/'

class ImageWorker:

    def __init__(self, training_dir, validation_dir): #CROSSWALIDACJA STRATYFIKOWANA!!

        self.history = None;
        self.model = None;

    def setDirs(self):

        train_generator = train_datagen.flow_from_directory(
          training_dir,
          target_size=(300, 300),
          class_mode='binary'
        )

        validation_generator = train_datagen.flow_from_directory(
          validation_dir,
          target_size=(300, 300),
          class_mode='binary'
        )

    def setNetworkParams(self):
        self.model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                      input_shape=(300, 300, 3)),
          tf.keras.layers.MaxPooling2D(2, 2),
          tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
          tf.keras.layers.MaxPooling2D(2,2),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(512, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.summary()

    def specifyNetworkParams(self):

        self.model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

    def trainModel(self):

        assert(self.model != None, f'It looks like you have not set the network parameters yet. Before training, call setNetworkParams() and specifyNetworkParams().')

        history = model.fit(
          train_generator,
          epochs=2
        )

    def validateModel(self):

        assert(self.model != None, f'It looks like you have not set the network parameters yet. Before training, call setNetworkParams() and specifyNetworkParams().')

        history = model.fit(
          train_generator,
          epochs=15,
          validation_data=validation_generator
        )
