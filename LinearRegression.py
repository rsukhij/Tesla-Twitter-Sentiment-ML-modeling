# Necessary imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sn


# Main function
def main():
    # Get dataframe from cleaned CSV of data
    df = pd.read_csv('total_engagement_filtered.csv')
    df['Date'] = pd.to_datetime(df.date)
    df['date'] = pd.to_datetime(df.date)
    df['date'] = df['date'].map(dt.datetime.toordinal)
    # X data
    sentiment_column = df.loc[:, 'compound']
    sentiment = sentiment_column.values

    # Y data
    close_price_column = df.loc[:, 'Percent change 1 week']
    close_price = close_price_column.values

    # Convert the data into numpy arrays
    x = np.array(sentiment).reshape((-1, 1))
    y = np.array(close_price)

    # Split data intro training and test arrays 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create the Linear Regression Model and fit it on the training data
    model = LinearRegression().fit(X_train, y_train)

    # Test it on the training data and get R^2 value to see goodness of fit
    r_sq = model.score(X_test, y_test)

    # Get predicted values
    y_pred = model.predict(X_test)

    # Performance metrics
    print("R squared value: ", r_sq)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Print out model intercept
    print("Model intercept: ", model.intercept_, "\n")

    # Print out model slope/coefficient
    print("Model slope: ", model.coef_, "\n")

    # Plot linear regression model
    # plt.scatter(df['compound'], df['Percent change 1 week'])

    # axes = plt.gca()
    # x_val = np.array(axes.get_xlim())
    # y_val = model.intercept_ + model.coef_ * x_val
    # plt.plot(x_val, y_val, '--', color='red')
    plt.scatter(X_test, y_test)
    plt.scatter(X_test, y_pred, color='red')
    plt.xlim([-1, 1])
    plt.title("Linear Regression Model")
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

    df_cm = pd.DataFrame(cm, range(2), range(2))
    sn.set(font_scale=1.5)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})
    plt.show()


if __name__ == '__main__':
    main()
