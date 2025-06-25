"""
visualization.py

Provides functions for visualizing data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import probplot
from sklearn.metrics import r2_score


def plot_model(x_train, y_train, x_test, y_test, x_params, y_param, model, scatter,
               remove_negatives=True, model_name=False, title=False):
    """
    Plots training and test predictions for a given model, handling optional
    scatter plots, negative values, and custom titles.
    """
    y_pred_train = model.predict(pd.DataFrame(x_train, columns=x_params))
    y_pred_test = model.predict(pd.DataFrame(x_test, columns=x_params))

    # Handle negative predictions
    if remove_negatives:
        y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
        y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)

    # Calculate R^2 scores
    r2_train = r2_score(y_train[y_param], y_pred_train)
    r2_test = r2_score(y_test[y_param], y_pred_test)

    # Plotting the results
    if not model_name:
        model_name = str(model).split("(")[0]

    plt.figure(figsize=(20, 5))

    # Training data plot
    plt.subplot(1, 2, 1)
    if scatter:
        plt.scatter(x_train.index, y_train[y_param], label="Reference", s=10)
        plt.scatter(y_train.index, y_pred_train, label="Prediction", s=10)
    else:
        plt.plot(x_train.index, y_train[y_param], label="Reference")
        plt.plot(y_train.index, y_pred_train, label="Prediction")
    plt.legend()
    plt.title(f"{model_name} result on training data (R²: {r2_train:.6f})")

    # Test data plot
    plt.subplot(1, 2, 2)
    if scatter:
        plt.scatter(x_test.index, y_test[y_param], label="Reference", s=10)
        plt.scatter(y_test.index, y_pred_test, label="Prediction", s=10)
    else:
        plt.plot(x_test.index, y_test[y_param], label="Reference")
        plt.plot(y_test.index, y_pred_test, label="Prediction")
    plt.legend()
    plt.title(f"{model_name} result on test data (R²: {r2_test:.6f})")
    if title:
        plt.suptitle(title, fontsize=18)

    plt.show()


def plot_GRU_model(x_train, y_train, train_dates, x_test, y_test, test_dates,
                   y_param, model, scatter, remove_negatives=True, title=False):
    """
    Plots GRU model predictions on training and test data over time, 
    with options for scatter plots, negative value handling, and custom titles.
    """
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    y_pred_train = y_pred_train.flatten()
    y_pred_test = y_pred_test.flatten()

    # Handle negative predictions
    if remove_negatives:
        y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
        y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)

    # Calculate R^2 scores
    r2_train = r2_score(y_train[y_param], y_pred_train)
    r2_test = r2_score(y_test[y_param], y_pred_test)

    # Training data plot
    model_name = "GRU model"
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    if scatter:
        plt.scatter(train_dates, y_train[y_param], label="Reference", s=10)
        plt.scatter(train_dates, y_pred_train, label="Prediction", s=10)
    else:
        plt.plot(train_dates, y_train[y_param], label="Reference")
        plt.plot(train_dates, y_pred_train, label="Prediction")
    plt.legend()
    plt.title(f"{model_name} - result on training data (R²: {r2_train:.6f})")

    # Test data plot
    plt.subplot(1, 2, 2)
    if scatter:
        plt.scatter(test_dates, y_test[y_param], label="Reference", s=10)
        plt.scatter(test_dates, y_pred_test, label="Prediction", s=10)
    else:
        plt.plot(test_dates, y_test[y_param], label="Reference")
        plt.plot(test_dates, y_pred_test, label="Prediction")
    plt.legend()
    plt.title(f"{model_name} - result on test data (R²: {r2_test:.6f})")
    if title:
        plt.suptitle(title, fontsize=18)

    plt.show()


def plot_losses(history):
    """
    Plots the training and validation loss over epochs from 
    a Keras model's training history.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss and validation loss
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.axhline(loss[-1], color="blue", alpha=0.3, linestyle="--")
    plt.axhline(val_loss[-1], color="red", alpha=0.3, linestyle="--")
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


def plot_residuals(residuals, prediction):
    """
    Plots residuals over time, residuals vs fitted values, histogram of 
    residuals, and a Q-Q plot to assess model performance.
    """
    plt.figure(figsize=(14, 10))

    # 1. Residuals Over Time
    plt.subplot(2, 2, 1)
    plt.plot(residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residuals')

    # 2. Residuals vs Fitted Plot
    plt.subplot(2, 2, 2)
    plt.scatter(prediction, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')

    # 3. Histogram of Residuals
    plt.subplot(2, 2, 3)
    sns.histplot(residuals, kde=True)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')

    # 4. Q-Q Plot
    plt.subplot(2, 2, 4)
    probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')

    plt.tight_layout()
    plt.show()
