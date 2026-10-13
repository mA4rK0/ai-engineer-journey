from sentence_transformers import SentenceTransformer
import json
import numpy as np

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

teks = "Apa itu RAG?"
vec = model.encode(teks)

print("\nKalimat :", teks)
print("Dimensi :", vec.shape)          
print("5 angka pertama:", vec[:5])     
print("Tipe data:", vec.dtype)

docs = []
with open("data/dataset_faq_rag_llm.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        docs.append(item["question"] + " " + item["answer"])

print(f"\nJumlah dokumen: {len(docs)}")
embeddings = model.encode(docs, show_progress_bar=True)

print("Shape embeddings:", embeddings.shape)

np.save("data/embeddings.npy", embeddings)
with open("data/docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("\nTersimpan: data/embeddings.npy + data/docs.json")

a = model.encode("Apa itu RAG?")
b = model.encode("Jelaskan cara kerja Retrieval-Augmented Generation.")
c = model.encode("Berapa harga cabai hari ini?")
def cos(u, v): return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
print("\nRAG vs RAG (parafrase):", round(cos(a, b), 3))   
print("RAG vs harga cabai     :", round(cos(a, c), 3))   