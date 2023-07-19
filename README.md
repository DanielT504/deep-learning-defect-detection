# Polytect: Deep Learning Defect Detection
Real-Time Classification of Polymer Insulator Defects using Deep Learning

Abstract:
Insulators are important to the integrity of the power grid. There are two main types: ceramic and non-ceramic (polymer) insulators. While the former are stable, their performance quickly degrades due to pollution, while the latter has a hydrophobic surface and can maintain higher performance for longer. Over time however, polymer insulators lose their hydrophobicity, which makes it very important to assess their surface conditions. The IEEE standards have developed a guide to classify the hydrophobicity from 1 to 7 and the classification received from this scale will depend on the expertise of the utility engineer. This is a slow process and expert utility engineers are not always available, which creates a need to automate the process of assessing the hydrophobicity. This project’s objective will be developing a phone app that uses deep learning to assist identifying the hydrophobicity class of the polymer surface, bridging the gap between hardware and software in the power grids industry. Through real-time image processing or photo uploads, the app will allow users to view what hydrophobicity classification level the polymer insulator currently has. Furthermore, numerous data sets exist to pre-train our deep learning model, allowing for generalized and non-bias results to be consistently shown.

PLEASE READ:
    The first time you run the code, it will produce approximately 8GB of preprocessed images in the preset folders.
    If you do not want this (for dev purposes), comment out the first line in cnn_model.py.
    If you do unpack the images, make sure not to commit the dataset folder, because it is way too large for a git repo.
    After the images are preprocessed, you won't need to wait for them next time.

TODO:
Next models to try: ResNet50 ResNet101 Xception InceptionV3 MobilenetV2 MobileNet
Merge 5, 5+ labels and update confusion matrix

Most recent evaluation (very good):
    Total training images used: 6074
    Total testing images used: 242
Epoch 1/10
190/190 [==============================] - 48s 163ms/step - loss: 0.7509 - accuracy: 0.7315
Epoch 2/10
190/190 [==============================] - 33s 173ms/step - loss: 0.2799 - accuracy: 0.9052
Epoch 3/10
190/190 [==============================] - 32s 170ms/step - loss: 0.1987 - accuracy: 0.9335
Epoch 4/10
190/190 [==============================] - 35s 184ms/step - loss: 0.1397 - accuracy: 0.9569
Epoch 5/10
190/190 [==============================] - 31s 165ms/step - loss: 0.1331 - accuracy: 0.9577
Epoch 6/10
190/190 [==============================] - 32s 167ms/step - loss: 0.1060 - accuracy: 0.9684
Epoch 7/10
190/190 [==============================] - 32s 169ms/step - loss: 0.0939 - accuracy: 0.9700
Epoch 8/10
190/190 [==============================] - 33s 171ms/step - loss: 0.0784 - accuracy: 0.9761
Epoch 9/10
190/190 [==============================] - 36s 190ms/step - loss: 0.1262 - accuracy: 0.9646
Epoch 10/10
190/190 [==============================] - 38s 196ms/step - loss: 0.0718 - accuracy: 0.9794
8/8 [==============================] - 3s 74ms/step - loss: 0.0220 - accuracy: 0.9959
Test Loss: 0.02202831394970417
Test Accuracy: 0.9958847761154175
8/8 [==============================] - 3s 38ms/step

Confusion Matrix:

<img width="789" alt="Screen Shot 2023-07-19 at 1 11 19 PM" src="https://github.com/DanielT504/deep-learning-defect-detection/assets/59990709/11d460da-5a33-4e48-bc6d-039a989dc90b">


WIP:
Mehr - EfficientNetB7
Claudia - MobilenetV2
