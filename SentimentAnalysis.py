# Necessary imports
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from yfinance import Ticker
import numpy as np


# Main function
def main():
    # Read in CSV of just date and tweets into a data frame
    df = pd.read_csv('elonmusk.csv')

    # Convert String dates into datetime data type
    df['date'] = pd.to_datetime(df.date)
    print(df.info())

    # Download lexicon to use for sentiment analysis
    nltk.download('vader_lexicon')

    # Create sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Create column in data frame for sentiment analysis scores
    df['scores'] = df['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

    # Create column in data frame for just compound scores
    df['compound'] = df['scores'].apply(lambda d: d['compound'])

    # Create column in data frame to classify sentiment as positive or negative based on the compound score
    df['score'] = df['compound'].apply(
        lambda score: 'pos' if score >= 0 else 'neg')

    # Output data frame as CSV
    df.to_csv("analyzed_tweets.csv")

    # Stock price code starts here

    # Create TSLA ticker object
    tkr = Ticker('TSLA')

    # Get 10y TSLA history
    hist = tkr.history(period="10y")

    # isolate date and closing price
    df_fi = hist.iloc[:, 3]

    # Read data from CSV created from df_fi dataframe previously
    stock = pd.read_csv('stockdata.csv')

    # # Convert String dates into datetime data type
    stock['date'] = pd.to_datetime(stock.date)

    # Merge the tweet, sentiment, and stock data into one dataframe by date
    merged = pd.merge(df, stock, how='outer', on='date')

    # Replace holiday/weekend dates with stock value from last trading day
    merged['Close'] = merged['Close'].replace('', np.nan).bfill()

    # Create CSV of this data
    merged.to_csv('final_data.csv')


if __name__ == '__main__':
    main()
