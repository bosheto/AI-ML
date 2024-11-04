import tensorflow as tf
import keras 

class TargetAccuracy(keras.callbacks.Callback):
    def __init__(self, target_accuracy):
        self.target = target_accuracy
        super().__init__()
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > self.target):
            print(f'\nReached target accuracy of {self.target * 100}%\nstopping model.')
            self.model.stop_training = True

callbacks = TargetAccuracy(0.95)
data = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

model.summary()