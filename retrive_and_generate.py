from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Load from .env

# Load model and FAISS index
model = SentenceTransformer("models/fine_tuned_model")
index = faiss.read_index("pipeline/quote_index.faiss")

with open("pipeline/index_texts.pkl", "rb") as f:
    texts = pickle.load(f)
with open("pipeline/index_meta.pkl", "rb") as f:
    metadata = pickle.load(f)

def retrieve_quotes(query, k=5):
    q_embedding = model.encode([query])
    distances, indices = index.search(np.array(q_embedding), k)
    return [(texts[i], metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]

def generate_response(query):
    retrieved = retrieve_quotes(query)
    
    context_str = "\n".join([f"{i+1}. \"{meta['quote']}\" — {meta['author']} ({', '.join(meta['tags'])})"
                             for i, (_, meta, _) in enumerate(retrieved)])
    
    prompt = f"""You are a quote assistant.
Query: {query}
Based on these quotes:
{context_str}

Provide a structured JSON response with quote, author, tags, and a brief summary.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or gpt-4
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response['choices'][0]['message']['content']

# Test
if __name__ == "__main__":
    print(generate_response("humorous quotes by Oscar Wilde"))
