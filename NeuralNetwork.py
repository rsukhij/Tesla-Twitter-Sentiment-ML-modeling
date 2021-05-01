import numpy as np
import pandas as pd
import seaborn as sn
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split


# Get the baseline model with one neuron for the one input attribute
def baseline_model():
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.add(Dense(1))

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

print(X_train.shape)

# Train the model with 100 epochs
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, verbose=0)
# estimator.fit(X_train, y_train, validation_split=0.2)

estimator = Sequential()
estimator.add(Dense(1, input_dim=1, activation='relu'))
estimator.add(Dense(1, activation='relu'))
estimator.add(Dense(1))

estimator.compile(loss='mean_squared_error', optimizer='adam')

history = estimator.fit(X_train, y_train, epochs=100, batch_size=5)

# Test results and metrics
y_pred = estimator.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R Squared Score is:', r2_score(y_test, y_pred))

# Plotting results
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred, color='red')
plt.xlim([-1, 1])
plt.title("Neural Network Model")
plt.xlabel("Sentiment")
plt.ylabel("Percent change in price by week")
plt.show()

# Confusion matrix
actual = [None] * len(y_test)
predicted = [None] * len(y_pred)
i = 0

for element in y_test:
    if element > 0:
        actual[i] = 1
    elif element == 0:
        actual[i] = 0
    elif element < 0:
        actual[i] = -1
    i += 1

i = 0
for element in y_pred:
    if element > 0:
        predicted[i] = 1
    elif element == 0:
        predicted[i] = 0
    elif element < 0:
        predicted[i] = -1
    i += 1

cm = confusion_matrix(actual, predicted)

print("Accuracy:", accuracy_score(actual, predicted))

df_cm = pd.DataFrame(cm, range(2), range(2))
sn.set(font_scale=1.5)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})
plt.show()
