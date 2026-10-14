import json
import numpy as np
from sentence_transformers import SentenceTransformer

docs = json.load(open("data/docs.json", encoding="utf-8"))
embs = np.load("data/embeddings.npy")
print(f"Loaded {len(docs)} dokumen, shape embeddings: {embs.shape}")

model = SentenceTransformer("all-MiniLM-L6-v2")
query = input("\nTanya: ")
q = model.encode(query)
print("Shape query:", q.shape)

scores = embs @ q

top3 = np.argsort(scores)[::-1][:3]

print("\n=== 3 dokumen paling relevan ===")
for rank, i in enumerate(top3, start=1):
    print(f"\n#{rank}  skor={scores[i]:.3f}")
    print(docs[i])
