import pandas as pd

keywords = ["tesla", "TSLA", "Tesla", "stock", "market", "stock market", "$TSLA"]
searched_keywords = '|'.join(keywords)
df = pd.read_csv("data_with_total_engagement.csv", sep=",")
df = df[df["tweet"].str.contains(searched_keywords)]
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['date', 'total_engagement'], ascending=True)
df = df.drop_duplicates(subset='date', keep="last")
df.to_csv("total_engagement_filtered.csv", sep=",", index=False)
