import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

# constants: current dimensions
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

# functions
def collect_image_paths(directory, excluded_folder, selected_image_paths_train = [], selected_image_paths_test = []):
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
    return num_images_used_train, num_images_used_test, selected_image_paths_train, selected_image_paths_test

def populate_formatted_image_array(selected_image_paths):
    image_array = []
    for image_path in selected_image_paths:
        image = Image.open(image_path)
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image.convert('RGB')
        image = np.array(image) / 255.0
        image_array.append(image)
    image_array = np.array(image_array)
    return image_array

def plot_confusion_matrix(test_labels, predicted_labels, num_classes):
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