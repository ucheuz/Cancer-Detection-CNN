# Cancer-Detection-CNN
A machine learning algorithm to detect cancerous cells in histopathological images using a convolutional neural network (CNN). The model processes 96x96 grayscale images and classifies them into cancerous (Label 1) or non-cancerous (Label 0) categories. The CNN architecture consists of two convolutional layers with ReLU activation, followed by fully connected layers for classification. The model is trained using cross-entropy loss and optimized with a suitable learning rate. Performance evaluation includes validation on a test set, accuracy measurement, and visualization of model performance over epochs. The project includes:
  A Python Script (main.py) that runs the model.
  A Juypter Notebook (Cancer Detection.ipynb) Task 4 with exploratory data analyis, model training and conclusions about the model at each stage.

# Description
In this project, I aimed to develop a machine learning model to detect cancerous cells in histopathological images using a convolutional neural network (CNN). The dataset consisted of 96x96 grayscale images, categorized into cancerous and non-cancerous classes. To extract meaningful features, I designed a CNN architecture with two convolutional layers (learning 16 and 32 filters, respectively) followed by two fully connected layers for classification. ReLU activation was applied to all hidden layers, and cross-entropy loss was used as the objective function. The model was trained using an optimized learning rate and validated on a test set to assess its generalization performance. The final implementation included performance evaluation through accuracy measurement, loss visualization over training epochs, and validation set assessment to ensure reliable classification of histopathological images.

# How to run
Option 1: Running the Python Script: Can be done directly in terminal. This will train the model and output predictions and accuracy scores.

Option 2: Running the Juypter Notebook: Download and open Cancer Detection.ipynb and run the cells in Task 4 step by step.

# Dataset
File: datasets/histological_data.npz
Description: A collection of 96x96 histological images.

# Results
Best Validation Accuracy: 0.8267

# Contributors
Kosiasochukwu Uchemudi Uzoka - Author

