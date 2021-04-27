# Necessary imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Main function
def main():
    # Get dataframe from cleaned CSV of data
    df = pd.read_csv('total_engagement_filtered.csv')

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
    print("R squared value: ", r_sq, "\n")

    # Print out model intercept
    print("Model intercept: ", model.intercept_, "\n")

    # Print out model slope/coefficient
    print("Model slope: ", model.coef_, "\n")

    # Plot linear regression model
    axes = plt.gca()
    x_val = np.array(axes.get_xlim())
    y_val = model.intercept_ + model.coef_ * x_val
    plt.plot(x_val, y_val, '--')
    plt.title("Linear Regression Model")
    plt.xlabel("Sentiment")
    plt.ylabel("Closing price")
    plt.show()


if __name__ == '__main__':
    main()
