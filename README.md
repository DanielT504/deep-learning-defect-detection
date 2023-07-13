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
Next models to try: EfficientNetB7 ResNet50 ResNet101 Xception InceptionV3 MobilenetV2 MobileNet
Merge 5, 5+ labels and update confusin matrix

Most recent evaluation (pretty bad):
    Total training images used: 7100
    Total testing images used: 284
    Epoch 1/10
    222/222 [==============================] - 57s 142ms/step - loss: 2.1286 - accuracy: 0.1408
    Epoch 2/10
    222/222 [==============================] - 32s 145ms/step - loss: 1.9821 - accuracy: 0.1514
    Epoch 3/10
    222/222 [==============================] - 32s 143ms/step - loss: 1.9519 - accuracy: 0.1390
    Epoch 4/10
    222/222 [==============================] - 32s 146ms/step - loss: 1.9472 - accuracy: 0.1501
    Epoch 5/10
    222/222 [==============================] - 32s 144ms/step - loss: 1.9530 - accuracy: 0.1496
    Epoch 6/10
    222/222 [==============================] - 32s 143ms/step - loss: 1.9467 - accuracy: 0.1513
    Epoch 7/10
    222/222 [==============================] - 32s 143ms/step - loss: 1.9474 - accuracy: 0.1480
    Epoch 8/10
    222/222 [==============================] - 32s 143ms/step - loss: 1.9476 - accuracy: 0.1527
    Epoch 9/10
    222/222 [==============================] - 31s 142ms/step - loss: 1.9461 - accuracy: 0.1504
    Epoch 10/10
    222/222 [==============================] - 32s 143ms/step - loss: 1.9446 - accuracy: 0.1518
    9/9 [==============================] - 2s 34ms/step - loss: 1.9519 - accuracy: 0.1162
    Test Loss: 1.9519131183624268
    Test Accuracy: 0.11619718372821808
    9/9 [==============================] - 2s 34ms/step