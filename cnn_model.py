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
selected_image_paths_train = []
selected_image_paths_test = []

def collect_image_paths(directory, excluded_folder):
    num_images_used_train = 0
    num_images_used_test = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            if os.path.basename(item_path).lower() != excluded_folder.lower():
                used_train, used_test = collect_image_paths(item_path, excluded_folder)
                num_images_used_train += used_train
                num_images_used_test += used_test
            else:
                _, used_test = collect_image_paths(item_path, excluded_folder)
                num_images_used_test += used_test
        else:
            if item_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                if excluded_folder not in os.path.dirname(item_path):
                    selected_image_paths_train.append(item_path)
                    num_images_used_train += 1
                else:
                    selected_image_paths_test.append(item_path)
                    num_images_used_test += 1
    return num_images_used_train, num_images_used_test

excluded_folder = "original_only"
num_images_used_train, num_images_used_test = collect_image_paths(root_dir, excluded_folder)

print(f"Total training images used: {num_images_used_train}")
print(f"Total testing images used: {num_images_used_test}")

train_images = []
for image_path in selected_image_paths_train:
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = image.convert('RGB')
    image = np.array(image) / 255.0
    train_images.append(image)

test_images = []
for image_path in selected_image_paths_test:
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = image.convert('RGB')
    image = np.array(image) / 255.0
    test_images.append(image)

train_images = np.array(train_images)
test_images = np.array(test_images)

image_height = 32
image_width = 32
num_channels = 3
num_classes = 7
num_epochs = 10
batch_size = 32

train_labels = np.random.randint(1, num_classes + 1, size=num_images_used_train) - 1
test_labels = np.random.randint(1, num_classes + 1, size=num_images_used_test) - 1

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
