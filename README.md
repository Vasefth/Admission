# Admission

## Overview

This project aims to predict the likelihood of a student being accepted into graduate school based on various application factors using deep learning techniques. By analyzing a dataset containing parameters such as test scores and other application factors, we employ TensorFlow with Keras to create a regression model. The goal is to gain insights into the graduate admissions process and to help prospective students improve their application strategies.

## Dataset

The dataset, `admissions_data.csv`, includes several features relevant to graduate school applications:

- GRE Scores
- TOEFL Scores
- University Rating
- SOP (Statement of Purpose)
- LOR (Letter of Recommendation Strength)
- CGPA (Undergraduate GPA)
- Research Experience
- Chance of Admit (Target Variable)

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

First, make sure Python 3 is installed on your system. Then, install the required libraries using pip:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Model Architecture

The regression model is designed as follows:

- An input layer that matches the number of features in the dataset.
- Two hidden layers with ReLU activation and dropout layers to prevent overfitting.
- An output layer with a single neuron for predicting the chance of admission.
- The model uses the Adam optimizer and Mean Squared Error (MSE) as the loss function.

## Training

The model is trained with the following strategy:

- The data is split into training and test sets (67% training, 33% test).
- Features are standardized using `StandardScaler`.
- Early stopping is implemented to halt training when the validation loss stops improving, preventing overfitting.
- The model is evaluated using the test set, with results reported in terms of MSE and Mean Absolute Error (MAE).

## Evaluation

The performance of the model is evaluated using:

- Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the test set.
- R-squared score to determine how well the regression model predicts the target variable.
- Plots of MAE and loss over epochs for both training and validation sets to visualize the learning process.

## Usage

Run the script to train the model and evaluate its performance. It will output the MSE, MAE, and R-squared score, along with plots showing the model's training history:

```bash
python admissions_prediction.py
```

## Contributing

Contributions to the project are welcome! Please fork the repository, make your changes, and submit a pull request.
