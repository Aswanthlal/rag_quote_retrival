from datasets import load_dataset
import pandas as pd

dataset=load_dataset("Abirate/english_quotes", split="train")
dataset
df=pd.DataFrame(dataset)

print(df.head())
print(df.columns)
print(df.isnull().sum())

df['quote']=df['quote'].str.strip().lower()
df['author']=df['author'].str.strip()
df['tags']=df['tags'].apply(lambda x:[tag.lower().str() for tag in x])

df.to_csv("data/processed/cleaned_quotes.csv", index=False)