# Image Classification with Gradient Boosting and Convolutional Neural Network (CNN)

Welcome to the Image Classification repository! This project focuses on utilizing Gradient Boosting and Convolutional Neural Networks (CNN) for image classification tasks. The primary goal is to classify images from the Chinese MNIST dataset obtained from Kaggle.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
  - [Gradient Boosting Classifier](#gradient-boosting-classifier)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Results](#results)
- [Contributing](#contributing)

## Introduction
Image classification is a fundamental task in computer vision that involves assigning labels to images based on their visual content. In this project, we explore two different approaches to tackle this problem - Gradient Boosting and Convolutional Neural Network (CNN). We start by applying a Gradient Boosting Classifier to the Chinese MNIST dataset, followed by employing a CNN using TensorFlow Keras for enhanced accuracy.

## Dataset
The dataset used in this project is the Chinese MNIST dataset, which is sourced from Kaggle. It contains a collection of images representing different Chinese characters. These images will serve as the basis for training and evaluating our image classification models.

## Models

### Gradient Boosting Classifier
In the initial phase of this project, we utilized the Gradient Boosting Classifier to perform image classification. The image files were extracted from a data folder, and the model was trained using this data. To optimize the classifier's performance, we employed GridSearchCV to find the best hyperparameters. However, the accuracy achieved by the Gradient Boosting Classifier was modest, peaking at 0.4.

### Convolutional Neural Network (CNN)
To achieve more accurate image classification, we implemented a Convolutional Neural Network (CNN) using TensorFlow's Keras library. The CNN architecture comprises several sequential layers designed to capture intricate patterns and features within the images. The model's progress was monitored by plotting the loss and accuracy curves. To fine-tune the CNN's performance, we employed Keras Tuner with Random Search, resulting in a significant improvement in accuracy. The final accuracy achieved by the CNN model was an impressive 0.94.

## Results
The CNN model outperformed the Gradient Boosting Classifier by a significant margin, achieving an accuracy of 0.94 compared to the 0.4 accuracy of the Gradient Boosting model. The repository contains Jupyter Notebooks illustrating each step of the process, from data extraction to the final trained models.

## Contributing
Contributions to this repository are welcome! If you find any issues or have ideas for improvements, feel free to open an issue or create a pull request. Your feedback and contributions will be highly appreciated.
