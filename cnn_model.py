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
from sklearn.metrics import confusion_matrix

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
num_images_to_select = 4000
random_image_paths = random.sample(selected_image_paths, num_images_to_select)

# Grainy and small, but functional
random_images = []
for image_path in random_image_paths:
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = image.convert('RGB')
    image = np.array(image) / 255.0
    random_images.append(image)

image_height = 32
image_width = 32
num_channels = 3
num_classes = 7
num_epochs = 10
batch_size = 32
train_images = np.array(random_images)
train_labels = np.random.randint(0, num_classes, size=num_images_to_select)
test_images = np.array(random_images)
test_labels = np.random.randint(0, num_classes, size=num_images_to_select)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_height, image_width, num_channels))
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size)

# Evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

# Predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Plot confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(1, num_classes + 1))
plt.yticks(tick_marks, range(1, num_classes + 1))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

thresh = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
