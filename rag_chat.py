import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

docs = json.load(open("data/docs.json", encoding="utf-8"))
embs = np.load("data/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Loaded {len(docs)} dokumen. RAG v0 siap — ketik 'exit' untuk keluar.")

THRESHOLD = 0.5

while True:
    query = input("\nTanya: ").strip()
    if query.lower() == "exit":
        print("Sampai jumpa!")
        break
    if not query:
        continue

    q = model.encode(query)
    scores = embs @ q
    top3 = np.argsort(scores)[::-1][:3]

    if scores[top3[0]] < THRESHOLD:
        print(f"\nJawaban: Maaf, jawaban tidak ada di dokumen saya. (skor terbaik {scores[top3[0]]:.3f} < {THRESHOLD})")
        continue

    context = "\n\n".join(docs[i] for i in top3)
    prompt = f"""Jawablah pertanyaan berikut HANYA berdasarkan konteks di bawah ini. Jika jawabannya tidak ada di konteks, katakan "Tidak ada di dokumen saya." Gunakan Bahasa Indonesia.

    Konteks:
    {context}

    Pertanyaan: {query}
    Jawaban:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False},
        timeout=120,
    )
    answer = response.json()["response"]

    print(f"\nJawaban (skor dokumen terbaik {scores[top3[0]]:.3f}):")
    print(answer)
    print(f"\n[Sumber: dokumen #{top3[0]+1}, #{top3[1]+1}, #{top3[2]+1}]")
