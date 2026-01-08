import numpy as np
from tensorflow.keras import layers , models , datasets
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf


# Load the MNist Data

( train_images , train_labels ) , (test_images , test_labels) = datasets.mnist.load_data()

#preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# building cnn model

model = models.Sequential()

# first conv layer

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# second
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# third
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(train_images , train_labels , epochs= 5 , batch_size = 32 , validation_data=(test_images , test_labels))

# evaluate the model on test data

test_loss , test_acc = model.evaluate(test_images , test_labels)
print(f"test loss: {test_loss} , test accuracy: {test_acc} ")

#make pred on test images

predictions = model.predict(test_images , batch_size = 32)

print(f"predictions for first test image : {predictions[0]} ")

plt.imshow(test_images[0].reshape(28, 28), cmap='Greys_r')
plt.title(f'Prediction label :  {predictions[0]}' )
plt.show()



















