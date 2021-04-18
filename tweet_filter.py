import pandas as pd

keywords = ["tesla", "TSLA", "Tesla"]
searched_keywords = '|'.join(keywords)
df = pd.read_csv("elonmusk.csv", sep=",")
df = df[df["tweet"].str.contains(searched_keywords)]
df.to_csv("elon_filtered.csv", sep=",", index=False)