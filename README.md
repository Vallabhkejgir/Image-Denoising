# README

## Overview
This repository contains code for training a UNet-based deep learning model to enhance low-resolution images to high-resolution images. The model is trained using pairs of low and high-resolution images and utilizes TensorFlow and Keras libraries for building and training the neural network. The code includes data preprocessing, model definition, training, and evaluation.

## Requirements
- Python 3.x
- OpenCV
- Numpy
- TensorFlow
- Keras
- Scikit-Image
- TQDM
- Scikit-Learn

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
    ```
2. **Install the required packages:**
    ```bash
    pip install opencv-python-headless numpy tensorflow keras scikit-image tqdm scikit-learn
    ```
3. **Provide path files to the images**

## Summary
This code provides a comprehensive approach to training a UNet model for image super-resolution. It includes all necessary steps from data loading and preprocessing to model training and evaluation. By following the steps outlined above, you can train the model on your dataset and evaluate its performance using PSNR.
