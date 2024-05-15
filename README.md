# CIFAR-10_Image_Classification_with_deep_learning:
This project demonstrates image classification on the CIFAR-10 dataset using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes.

## Overview:
The goal of this project is to classify images into one of ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Two different neural network architectures are employed to achieve this task: ANN and CNN. The project highlights the comparative performance of these models on the CIFAR-10 dataset.

## Process:
### 1. Data Loading and Preprocessing
+ Loading the Data: The CIFAR-10 dataset is loaded using TensorFlow's Keras API. | Training set: 50,000 images. | Test set: 10,000 images.
+ Reshaping and Normalizing the Data: Labels are reshaped to a flat vector. | Images are normalized to have pixel values between 0 and 1.
+ Data Visualization: Sample images from the dataset are visualized along with their corresponding labels.

### 2. Building and Training the Models: Artificial Neural Network (ANN)
#### Model Architecture: 
+ A sequential model with three dense layers.
+ First layer: Flattens the input image. 
+ Hidden layers: Two dense layers with 3000 and 1000 neurons respectively, using ReLU activation.
+ Output layer: A dense layer with 10 neurons and sigmoid activation.
#### Compilation:
+ Optimizer: Stochastic Gradient Descent (SGD).
+ Loss Function: Sparse Categorical Cross-Entropy.
+ Metrics: Accuracy.
#### Training:
+ The ANN model is trained for 5 epochs on the training dataset.

### Convolutional Neural Network (CNN)
#### Model Architecture:
+ A sequential model with two convolutional layers followed by pooling layers and dense layers.
+ Convolutional layers: 32 and 64 filters respectively, with ReLU activation.
+ Pooling layers: MaxPooling with a 2x2 filter.
+ Dense layers: A fully connected layer with 64 neurons and ReLU activation.
+ Output layer: A dense layer with 10 neurons and softmax activation.
#### Compilation:
+ Optimizer: Adam.
+ Loss Function: Sparse Categorical Cross-Entropy.
+ Metrics: Accuracy.
#### Training:
+ The CNN model is trained for 20 epochs on the training dataset.

### 3. Evaluation and Outcome:
+ ANN Performance: Achieved an accuracy of approximately 45% on the test dataset. | The classification report provides precision, recall, and F1-score for each class.

+ CNN Performance: Achieved an accuracy of approximately 67% on the test dataset. | The classification report provides precision, recall, and F1-score for each class.

Conclusion
The CNN outperformed the ANN in terms of accuracy and other classification metrics, highlighting its effectiveness in image classification tasks due to its ability to capture spatial hierarchies in images. ANN can serve as a simpler baseline model, but for more complex image data, CNNs are generally preferred.
