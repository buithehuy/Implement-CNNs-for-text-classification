# Implement-CNNs-for-text-classification

This project demonstrates sentiment analysis of IMDB movie reviews using a Text Convolutional Neural Network (TextCNN) implemented in PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)


## Introduction

This project aims to classify movie reviews as either positive or negative using a TextCNN model. It involves preprocessing the text data, building and training the model, and evaluating its performance.

## Dataset

The project utilizes the "IMDB Dataset.csv" file containing movie reviews and their corresponding sentiments (positive or negative). The dataset is preprocessed by tokenizing the reviews, building a vocabulary, and converting the text into numerical indices.

## Model

The TextCNN model consists of an embedding layer, convolutional layers with different kernel sizes, a max-pooling layer, and a fully connected layer for classification. The model architecture is designed to capture local patterns and features in the text data for sentiment analysis.

## Usage

1.  **Data Preparation:** Make sure you have the "IMDB Dataset.csv" file in the same directory as the code.
2.  **Dependencies:** Install the necessary libraries using `pip install -r requirements.txt`.
3.  **Training:** Run the Python script to train the TextCNN model. The trained model will be saved as "text\_cnn\_model.pth".
4.  **Prediction:** Use the `predict_sentiment` function to predict the sentiment of a new movie review. Provide the input text and the trained model path.


## Results

Epoch 5/5, Loss: 0.2140
Test Accuracy: 0.8796

### test:
Enter a movie review: This movie is boring and violent!
Predicted Sentiment: Negative

## Dependencies

-   Python 3.x
-   pandas
-   numpy
-   torch
-   scikit-learn
-   tqdm
