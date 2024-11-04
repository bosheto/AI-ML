import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.optimizers import RMSprop
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
from keras._tf_keras.keras.preprocessing import image
import urllib
import numpy as np
import os 


model_path = 'models/binary-classifier-cats-and-dogs.keras'
training_dir = 'data/dogs-vs-cats/training/'

# weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

weights_file = "inception_v3.h5"
# urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(input_shape=(150,150,3), include_top=False, weights=None)

pre_trained_model.load_weights(weights_file)

# pre_trained_model.summary()


#All images will be rescaled by 1./255
# train_datagen = ImageDataGenerator(
#   rescale=1./255,
#   rotation_range=40,
#   width_shift_range=0.2,
#   height_shift_range=0.2,
#   shear_range=0.2,
#   zoom_range=0.2,
#   horizontal_flip=True,
#   fill_mode='nearest'
# )

# validation_datagen = ImageDataGenerator(rescale=1/255)
# validation_dir = 'data/horse-or-human/validation/'

train_ds = keras.utils.image_dataset_from_directory(
    training_dir,
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(150,150),
    shuffle=True
)

val_size = int(0.2 * len(train_ds))
train_ds = train_ds.skip(val_size)
val_ds = train_ds.take(val_size)

print(len(train_ds))
print(len(val_ds))

normalization_layer = keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# train_generator = train_datagen.flow_from_directory(
#     training_dir,
#     target_size=(150, 150),
#     class_mode='binary',
#     save_format='jpg'
# )

# validation_generator = train_datagen.flow_from_directory(
#   validation_dir,
#   target_size=(150, 150),
#   class_mode='binary'
# )

def train_model(epochs):
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output)
    last_output = last_layer.output

    x = keras.layers.Flatten()(last_output)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs=pre_trained_model.input, outputs=x)

    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=['acc'])

    model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    model.save(model_path)


def test_model():
    model = keras.models.load_model(model_path)

    img = image.load_img('data/dogs-vs-cats/test1/1.jpg', target_size=(150,150))
    y = image.img_to_array(img)
    y = np.expand_dims(y,axis=0)

    images = np.vstack([y])
    classes = model.predict(images, batch_size=10)
    if classes[0]>0.5:
        print('Dog')
    else:
        print('Cat')


train_model(15)
test_model()

# model.summary()

# model.compile(loss='binary_crossentropy',
#               optimizer=RMSprop(learning_rate=0.001),
#               metrics=['accuracy'])

# history = model.fit(train_generator, epochs=15, 
#                     validation_data=validation_generator)