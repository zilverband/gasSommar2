"""
model_training.py

Contains functions for training and evaluating different models.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from . import visualization as vz


def lin_reg(x_train, y_train, x_test, y_test, x_params, y_param, plot=False, scatter=False):
    """
    Fits a linear regression model to the training data, evaluates it on the
    test data, and optionally plots the results.
    """

    model = LinearRegression()
    model.fit(pd.DataFrame(x_train[x_params]), y_train[y_param])
    y_pred = model.predict(pd.DataFrame(x_test[x_params]))

    mse = mean_squared_error(y_test[y_param], y_pred)
    r2 = r2_score(y_test[y_param], y_pred)

    if plot:
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        print(f"coeffs: {model.coef_}\nintercept: {model.intercept_}")
        vz.plot_model(x_train, y_train, x_test, y_test, x_params, y_param, model, scatter)

    return model, r2


def xgb_model(x_train, y_train, x_test, y_test, x_params, y_param, plot=False, scatter=False):
    """
    Trains an XGBoost regression model using grid search, evaluates it on the 
    test data, and optionally plots the results.
    """
    param_grid = {
    'n_estimators': [150],      # Number of trees
    'learning_rate': [0.12],    # Step size shrinkage
    'max_depth': [2],           # Maximum depth of a tree
    'min_child_weight': [1.1],  # Minimum sum of instance weight (hessian) needed in a child
    'subsample': [0.7],         # Subsample ratio of the training instances
    'colsample_bytree': [0.8],  # Subsample ratio of columns when constructing each tree
    'reg_alpha': [1.5],         # Regularization constant
    'reg_lambda': [1.5]         # Regularization constant
    }

    # Initialize the model
    model = XGBRegressor(objective='reg:squarederror')

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=2, scoring='r2', n_jobs=1, verbose=3)
    grid_search.fit(pd.DataFrame(x_train[x_params]), y_train[y_param])

    # Predict and evaluate
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(pd.DataFrame(x_test[x_params]))
    y_pred = np.where(y_pred < 0, 0, y_pred)

    mse = mean_squared_error(y_test[y_param], y_pred)
    r2 = r2_score(y_test[y_param], y_pred)

    if plot:
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        print(f'Best Parameters: {grid_search.best_params_}')
        vz.plot_model(x_train, y_train, x_test, y_test, x_params, y_param, best_model, scatter)

    return best_model, r2


def rnd_forest(x_train, y_train, x_test, y_test, x_params, y_param, plot=True, scatter=False,
               title=False, n_estimators=500, random_state=60, min_samples_leaf=5, 
               save_model=False, model_filename="models/LKPG_Strandv_1/NO2-B43F/rnd_forest_test.pkl"):
    """
    Trains a random forest regression model using grid search, evaluates it on the 
    test data, optionally plots results, and saves the model if specified.
    """

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  random_state=random_state,
                                  min_samples_leaf=min_samples_leaf)
    param_grid = {
        'n_estimators': [500],
        'random_state': [60],
        'min_samples_leaf': [5]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, scoring='r2', n_jobs=1, verbose=3)
    grid_search.fit(pd.DataFrame(x_train[x_params]), y_train[y_param])

    # Predict and evaluate
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(pd.DataFrame(x_test[x_params]))

    mse = mean_squared_error(y_test[y_param], y_pred)
    r2 = r2_score(y_test[y_param], y_pred)

    if plot:
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        print(f'Best Parameters: {grid_search.best_params_}')
        vz.plot_model(x_train, y_train, x_test, y_test, x_params,
                      y_param, best_model, scatter, title=title)

    if save_model:
        joblib.dump(best_model, model_filename)
        print(f"Model saved to filename {model_filename}")

    return best_model, r2


def neural_network_model(x_train, y_train, x_test, y_test, x_params, y_param,
                         plot=False, scatter=False, learning_rate=0.01,
                         epochs=15, batch_size=10, validation_split=0.2):
    """
    Trains a neural network regression model on scaled data, evaluates it on the 
    test data, optionally plots the results, and adjusts learning parameters during training.
    """

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train),
                           columns=x_train.columns, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

    # Define the model
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.001)
    history = model.fit(x_train[x_params], y_train[y_param], callbacks=[early_stopping, reduce_lr],
                        epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    y_pred = model.predict(x_test[x_params])

    # Calculate residuals
    y_true = y_test[y_param].values
    residuals = pd.Series(y_true - y_pred.flatten(), index=y_test.index)

    mse = mean_squared_error(y_test[y_param], y_pred)
    r2 = r2_score(y_test[y_param], y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    if plot:
        vz.plot_model(x_train, y_train, x_test, y_test, x_params, y_param,
                      model, scatter, True, "Neural Network")
        vz.plot_losses(history)
        vz.plot_residuals(residuals, y_pred)

    return model, r2


def gru_model(x_train, y_train, train_dates, x_test, y_test, test_dates,
              y_param, model, plot=False, scatter=False,
              epochs=50, batch_size=128, validation_split=0.2, title=False):
    """
    Trains a GRU model, evaluates it on the test data, optionally plots 
    the results, and adjusts learning parameters during training.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.001)
    history = model.fit(x_train, y_train, callbacks=[early_stopping, reduce_lr],
                        epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    # Predict and calculate residuals
    y_pred = model.predict(x_test)
    residuals = y_test - y_pred.flatten()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.3f}')
    print(f'R^2 Score: {r2:.3f}')

    if plot:
        # Convert arrays to pandas DataFrame with correct column names
        y_train_df = pd.DataFrame(y_train, columns=[y_param])
        y_test_df = pd.DataFrame(y_test, columns=[y_param])
        y_pred_train_df = pd.DataFrame(model.predict(x_train).flatten(), columns=[y_param])
        y_pred_test_df = pd.DataFrame(y_pred.flatten(), columns=[y_param])

        # Ensure indices match
        y_train_df.index = range(len(y_train_df))
        y_test_df.index = range(len(y_test_df))
        y_pred_train_df.index = range(len(y_pred_train_df))
        y_pred_test_df.index = range(len(y_pred_test_df))

        vz.plot_GRU_model(x_train, y_train_df, train_dates, x_test, y_test_df,
                       test_dates, y_param, model, scatter, True, title)
        vz.plot_losses(history)
        vz.plot_residuals(residuals, y_pred.flatten())

    return y_pred, r2
