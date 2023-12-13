# Hand Gesture Recognition Using CNN 📈

## Introduction

This repository hosts a deep learning project focused on recognizing hand gestures representing numbers 0-9 in sign language. The project utilizes a Convolutional Neural Network (CNN) to accurately identify hand gestures from a dataset of 15,000 labeled images.

## Project Structure

- `HandGestureCNN.py`: Defines the CNN architecture used for gesture recognition.
- `Train.py`: Contains the code to train the CNN on the hand gesture dataset.
- `hand_gesture_model.pth`: Saved weights of the trained CNN model.
- `UI.py`: A user interface script to interact with the trained model and make predictions.
- `requirements.txt`: Lists all the Python dependencies required for the project.

## Dataset

The dataset comprises 15,000 labeled images representing numbers 0-9 in sign language. Each image in the dataset is pre-processed and labeled with the corresponding number it represents.

## CNN Architecture

The CNN model, defined in `HandGestureCNN.py`, is structured as follows:

- **Convolutional Layers**: Multiple layers designed to capture the hierarchical patterns in the images. Each layer uses filters to learn increasingly complex features.
- **Pooling Layers**: These layers reduce the spatial dimensions (width and height) of the input volume for the layers that follow.
- **Fully Connected Layers**: Dense layers that interpret the features extracted by the convolutional and pooling layers to perform classification.
- **Dropout**: Used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
- **Activation Functions**: ReLU (Rectified Linear Unit) used for adding non-linearity to the model, enabling it to learn more complex patterns.

## Training the Model

The training process, implemented in `Train.py`, involves several steps:

- **Data Preprocessing**: Images are resized, converted to grayscale, and normalized.
- **Model Training**: The model is trained using the preprocessed images, employing techniques like batch normalization and dropout for better generalization.
- **Optimization**: The Adam optimizer is used for adjusting the weights, with a learning rate scheduler to improve training efficiency.
- **Loss Function**: Cross-Entropy Loss is used as the loss function, suitable for multi-class classification tasks.

## Using the Model

To use the trained model for predicting hand gestures:

1. Ensure you have Python installed and set up on your system.
2. Install the necessary dependencies: `pip install -r requirements.txt`.
3. Run `UI.py` to start the user interface for interacting with the model.

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/KristiDodaj/HandGestureCNN.git
cd HandGestureCNN
pip install -r requirements.txt
