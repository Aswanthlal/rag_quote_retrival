from sentence_transformers import SentenceTransformer,InputExample,losses
from torch.utils.data import DataLoader
import pandas as pd
import os

df=pd.read_csv("data/processed/cleaned_quotes.csv")

examples=[]
for i, row in df.iterrows():
    query=f"{','.join(row['tags'])} quotes by {row['author']}"
    quote=row['quote']
    examples.append(InputExample(texts=[query,quote]))

model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

train_dataloader=DataLoader(examples,shuffle=True,batch_size=16)

train_loss=losses.CosineSimilarityLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,  # Set 3–5 for real use
    warmup_steps=100,
    output_path="models/fine_tuned_model"
)