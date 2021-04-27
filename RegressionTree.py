import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
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

# Create the Decision Tree Regressor using mean squared error as the splitting criteria and a max depth of 5
regressor = DecisionTreeRegressor(criterion='mse', max_depth=10)

# Creating the model and fitting it on the training data
model = regressor.fit(X_train, y_train)

# Predicted values from the model on the test data
y_pred = model.predict(X_test)

# Performance metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared Score is:', r2_score(y_test, y_pred))

