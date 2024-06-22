# Emotion Detection Using CNN and FER-2013 Dataset
## Overview
This project focuses on detecting human emotions using Convolutional Neural Networks (CNNs) trained on the FER-2013 dataset. By leveraging advanced CNN architectures like ResNet50v2 and VGG16, and employing techniques to handle class imbalance, the project aims to classify emotions accurately and deploy the model for real-time detection in live video streams.\
Deployed the model using Gradio in Hugging faces: [link](https://huggingface.co/spaces/himavanth508/Emotion_Detection)
## Features
Data Augmentation and Class Balancing: Addressed class imbalance in the FER-2013 dataset with image augmentation and class weights to enhance model robustness.
Advanced CNN Architectures: Designed and iterated on custom CNN models, optimizing performance with architectures such as VGG16 and ResNet50v2.
High Accuracy: Achieved 66% overall accuracy in emotion classification, with detailed performance metrics including precision, recall, and F1-scores for 7 emotion labels.
Real-Time Emotion Detection: Deployed the final model using Gradio and OpenCV, enabling dynamic on-screen emotion labels in live video streams.

## Tech Stack
Programming Language: Python

Frameworks: TensorFlow, Keras

Models: ResNet50v2, VGG16

Libraries: Numpy, Pandas, Matplotlib, OpenCV, Gradio

Kaggle link for the dataset I have used in this project: https://www.kaggle.com/datasets/msambare/fer2013

