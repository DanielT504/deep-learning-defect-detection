from preprocessing import preprocess_images
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from imgaug import augmenters as iaa


#stand-in MNIST datasets
from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# need to be adjusted
image_height = 28
image_width = 28
num_channels = 1
num_classes = 10
num_epochs = 10
batch_size = 32

# train_images = 
# train_labels = 
# test_images = 

model = Sequential()

#convolutional layer, more needed? adjust kernel size, strides, and filters
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)))

#more pooling needed?
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#fully connected layers, need more? add a dropout layer?
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size)

#evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

predictions = model.predict(test_images)

#plot
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_images[i].reshape(image_height, image_width), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels[i])
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis('off')
plt.show()