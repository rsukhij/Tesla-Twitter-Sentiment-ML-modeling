import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Get the baseline model with one neuron for the one input attribute
def baseline_model():
    model = Sequential()
    model.add(Dense(1, input_dim=1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Get dataframe from cleaned CSV of data
df = pd.read_csv('total_engagement_filtered.csv')

# X data is the compound scores
X = df.iloc[:, 8].copy()

# Y data is the percent change week by week
y = df.iloc[:, 10].copy()

# Split data intro training and test arrays 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model with 100 epochs and a batch size of 32
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
estimator.fit(X_train, y_train)

# Test results and metrics
y_pred = estimator.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R Squared Score is:', estimator.score(X_test, y_test))

# Plotting results
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred, color='red')
plt.xlim([-1, 1])
plt.title("Neural Network Model")
plt.xlabel("Sentiment")
plt.ylabel("Percent change in price by week")
plt.show()




