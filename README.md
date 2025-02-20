# Deep Learning Defect Detection
Real-Time Classification of Polymer Insulator Defects using Convolutional Neural Networks and Deep Learning

Abstract:
Polymer insulators play a vital role in maintaining our power grid integrity, but their hydrophobicity (resistance to water damage) deteriorates over time, impacting their performance. The conventional process of classifying the hydrophobicity level based on IEEE standards (a scale of 1 to 6) usually relies on the expert analysis of utility engineers, which is slow and not always readily available. To address this, we are introducing a cost-effective and novel approach that uses deep learning to automate the classification process. Our mobile and web apps employ a convolutional neural network on images either captured in real-time or mass-uploaded. Then our deep learning model, which has been pre-trained extensively on existing data, will identify the hydrophobicity grade with a disclosed level of confidence containing an accuracy and loss rating. Our CNN uses EfficientNetB0 architecture and has been trained on augmented datasets including greyscale, rotation, scaling, contrast, flipping, and Gaussian noise, as well as sequential combinations. These preprocessing techniques have improved the generalization of our model and minimized its bias in hopes of providing a more efficient, reliable, and safe solution that reduces dependency on human expertise.

Executive summary:
In power grids, polymer insulators leverage hydrophobic surfaces to operate; as a result, their effectiveness diminishes over time. This project focuses on developing a convolutional neural network that uses deep learning to classify these insulators, and then categorize them into the IEEE-defined levels of hydrophobicity ranging from 1 (high) to 6 (low). With augmented training datasets and select preprocessing techniques, the EfficientNetB0-based model will be able to provide timely, accurate assessments via web and mobile applications, lowering the barrier of entry to maintaining such insulators.

PLEASE READ:
    The first time you run cnn_model.py, it will produce approximately 8GB of preprocessed images in the preset folders.
    If you do not want this (for dev purposes), comment out the first line in cnn_model.py.
    If you do unpack the images, make sure not to commit the dataset folder because it is way too large for a git repo.
    After the images are preprocessed, you won't need to wait for them next time.
    Once the model is created and saved, it can be reused until recreated, and can be evaluated by running model_evaluation.py.
    Alternatively, use the existing keras model from our repo to run model_evaluation.py right away and produce a confusion matrix.

NEW:
    Differerent colored datasets (DG Dark Grey and LG Light Grey) have been introduced to the training and evaluation.
    Since there are far more of these training images than the indoor/outdoor sets, they have not been preprocessed.
    Our model also has no need for validation, so those images will be added to the test set.
    Unfortunately the originals (~100Mb) must be pulled directly from our repo instead of the OneDrive folder because the
    file hierarchy has been modified to maintain consistency.
    To retrain using only indoor/outdoor or only LG/DG, you should be able to just remove the unused training and test
    folders from the dataset (although this hasn't been tested yet).
    Training/testing accuracy/loss results and the confusion matrix for the full indoor/outdoor/DG/LG dataset can be found
    after the following results, which were trained and tested on only the indoor/outdoor datasets.


INDOOR/OUTDOOR:

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

<img width="789" alt="conf_matrix" src="https://github.com/DanielT504/deep-learning-defect-detection/assets/62156098/641edcf6-410e-4857-99da-f447fc7d5e40">


INDOOR/OUTDOOR/DG/LG:

    Epoch 1/10
    421/421 [==============================] - 75s 127ms/step - loss: 0.8244 - accuracy: 0.6890
    Epoch 2/10
    421/421 [==============================] - 54s 129ms/step - loss: 0.4553 - accuracy: 0.8389
    Epoch 3/10
    421/421 [==============================] - 54s 128ms/step - loss: 0.3513 - accuracy: 0.8744
    Epoch 4/10
    421/421 [==============================] - 54s 128ms/step - loss: 0.2896 - accuracy: 0.8952
    Epoch 5/10
    421/421 [==============================] - 54s 128ms/step - loss: 0.2641 - accuracy: 0.9102
    Epoch 6/10
    421/421 [==============================] - 54s 128ms/step - loss: 0.2309 - accuracy: 0.9212
    Epoch 7/10
    421/421 [==============================] - 54s 128ms/step - loss: 0.2242 - accuracy: 0.9250
    Epoch 8/10
    421/421 [==============================] - 54s 128ms/step - loss: 0.1767 - accuracy: 0.9381
    Epoch 9/10
    421/421 [==============================] - 54s 128ms/step - loss: 0.1738 - accuracy: 0.9416
    Epoch 10/10
    421/421 [==============================] - 54s 128ms/step - loss: 0.1727 - accuracy: 0.9428

    23/23 [==============================] - 3s 28ms/step - loss: 0.3302 - accuracy: 0.8847
    Test Loss: 0.33023664355278015
    Test Accuracy: 0.8847222328186035
    23/23 [==============================] - 2s 28ms/step

![conf-matrix2](https://github.com/DanielT504/deep-learning-defect-detection/assets/62156098/e8b8da1d-9c3c-4b2f-b8f5-c6116f5c16c8)
