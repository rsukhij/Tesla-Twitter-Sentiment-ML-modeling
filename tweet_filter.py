import pandas as pd

keywords = ["tesla", "TSLA", "Tesla", "stock", "market", "stock market", "$TSLA"]
searched_keywords = '|'.join(keywords)
df = pd.read_csv("elonmusk.csv", sep=",")
df = df[df["tweet"].str.contains(searched_keywords)]
df = df.sort_values(['date','likes_count'],ascending=True)
df = df.drop_duplicates(subset='date', keep="last")
df.to_csv("elon_filtered.csv", sep=",", index=False)