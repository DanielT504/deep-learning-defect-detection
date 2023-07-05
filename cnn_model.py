from preprocessing import preprocess_images
import os
import random
from efficientnet.keras import EfficientNetB0
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from imgaug import augmenters as iaa
from PIL import Image

root_dir = 'dataset'
selected_image_paths = []

def collect_image_paths(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            collect_image_paths(item_path)
        else:
            if item_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                selected_image_paths.append(item_path)

collect_image_paths(root_dir)
num_images_to_select = 10
random_image_paths = random.sample(selected_image_paths, num_images_to_select)

#grainy and small, but functional
random_images = []
for image_path in random_image_paths:
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = image.convert('RGB')
    image = np.array(image) / 255.0
    random_images.append(image)

'''
#stand-in MNIST datasets
from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
resized_train_images = []
resized_test_images = []
for img in train_images:
    resized_img = Image.fromarray(img).resize((32, 32)).convert('RGB')
    resized_train_images.append(np.array(resized_img))
for img in test_images:
    resized_img = Image.fromarray(img).resize((32, 32)).convert('RGB')
    resized_test_images.append(np.array(resized_img))
train_images = np.array(resized_train_images)
test_images = np.array(resized_test_images)
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
'''

image_height = 32
image_width = 32
num_channels = 3
num_classes = 10
num_epochs = 10
batch_size = 32
train_images = np.array(random_images)
train_labels = to_categorical(np.zeros(num_images_to_select), num_classes)
test_images = np.array(random_images)
test_labels = to_categorical(np.zeros(num_images_to_select), num_classes)

# train_images = 
# train_labels = 
# test_images = 

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_height, image_width, num_channels))
model = Sequential()
model.add(base_model)

'''
#convolutional layer, more needed? adjust kernel size, strides, and filters
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)))

#more pooling needed?
model.add(MaxPooling2D(pool_size=(2, 2)))
'''

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
for i in range(len(train_images)):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[i], cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(train_labels[i])
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis('off')
plt.show()