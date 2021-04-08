# Necessary imports
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import yfinance as yf
from datetime import date, timedelta, datetime

# Main function


def main():
    # Read in CSV of just date and tweets into a data frame
    df = pd.read_csv('elonmusk.csv')

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
    tkr = yf.Ticker('TSLA')

    # Get 10y TSLA history
    hist = tkr.history(period="10y")

    #isolate date and closing price
    df_fi = hist.iloc[:, 3]
    
    print(df_fi.head())

    #convert the date format of the dataset to match the closing price history dataset
    df['date'] = df['date'].apply(lambda date: datetime.strftime(datetime.strptime(date,'%m/%d/%Y'),'%Y-%m-%d'))
    
    print(df_fi.head())
    print(df_fi["Date"])

    #convert Date column to 
    df_fi["Date"] =  df_fi["Date"].apply(lambda Date: Date)

    closing_dict = df_fi.to_dict()


if __name__ == '__main__':
    main()
