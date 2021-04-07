# Necessary imports
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


# Main function
def main():
    # Read in CSV of just date and tweets into a data frame
    df = pd.read_csv('justtweets.csv')

    # Download lexicon to use for sentiment analysis
    nltk.download('vader_lexicon')

    # Create sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Create column in data frame for sentiment analysis scores
    df['scores'] = df['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

    # Create column in data frame for just compound scores
    df['compound'] = df['scores'].apply(lambda d: d['compound'])

    # Create column in data frame to classify sentiment as positive or negative based on the compound score
    df['score'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')
    print(df)

    # Output data frame as CSV
    df.to_csv("analyzed_tweets.csv")


if __name__ == '__main__':
    main()
