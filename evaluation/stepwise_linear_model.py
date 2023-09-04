import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector


def train_step_wise_linear_model(train_set, train_labels):
    train_set  = pd.DataFrame(train_set.numpy())
    train_labels = pd.DataFrame(train_labels.numpy())

    # neural network model with sklearn with 92 input, 40 hidden layer, 1 output
    # model = MLPRegressor(hidden_layer_sizes=(40,), activation='relu', solver='adam', max_iter=5000, learning_rate=0.01, alpha=0.001)

    for num_features in [20, 40, 60, 80, 92]:
        sfs = SequentialFeatureSelector(linear_model.LinearRegression(),
                                        k_features=num_features,
                                        forward=True,
                                        scoring='neg_mean_squared_error',
                                        cv=5)
        
        X_train, X_test,\
            y_train, y_test = train_test_split(
                train_set, train_labels,
                test_size=0.2,
                random_state=42)
        
        sfs.fit(X_train, y_train)
        selected_columns = train_set.columns[list(sfs.k_feature_idx_)]
        X_train = X_train[selected_columns]
        X_test = X_test[selected_columns]
        
        # Fit a logistic regression model using the selected features
        linreg = linear_model.LinearRegression()
        linreg.fit(X_train, y_train)
        # Make predictions using the test set
        y_pred = linreg.predict(X_test)
        print('number of features:', num_features, 'Mean squared error: %.2f', mean_squared_error(y_test, y_pred))


    # Evaluate the model performance
    # print(y_pred)

    