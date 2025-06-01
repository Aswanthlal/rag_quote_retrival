from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import os
import pickle

model=SentenceTransformer("models/fine_tuned_model")
df=pd.read_csv("data/processed/cleaned_quotes.csv")

texts=df.apply(lambda row: f'{row['quote']}-{row['author']} ({','.join(row['tags'])})', axis=1).tolist()
embddings=model.encode(texts, convert_to_numpy=True)

dimension=embddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(embddings)

faiss.write_index(index, "pipeline/quote_index.faiss")
with open("pipeline/index_texts.pkl", "wb") as f:
    pickle.dump(texts, f)
with open("pipeline/index_meta.pkl", "wb") as f:
    pickle.dump(df.to_dict('records'), f)

print("Index created and saved!")