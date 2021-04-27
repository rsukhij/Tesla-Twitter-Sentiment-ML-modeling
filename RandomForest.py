import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score

# Get dataframe from cleaned CSV of data
df = pd.read_csv('total_engagement_filtered.csv')

# X data is the compound scores
compound = df.iloc[:, 8].copy()

# Y data is the percent change week by week
percent_change = df.iloc[:, 10].copy()

# Convert the data into numpy arrays
X = np.array(compound).reshape((-1, 1))
y = np.array(percent_change)

# Split data intro training and test arrays 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Double check data formatting
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# Create the Random Forest Regressor with 1000 decision trees
regressor = RandomForestRegressor(n_estimators=1000, random_state=42)

# Fit the model using the training data
regressor.fit(X_train, y_train)

# Make predictions using the test data
y_pred = regressor.predict(X_test)

# Performance metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared Score is:', r2_score(y_test, y_pred))
