import json
import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rerank import rerank

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
docs = json.load(open("data/docs.json", encoding="utf-8"))
embs = np.load("data/embeddings.npy")
print(f"Siap melayani {len(docs)} dokumen.")

THRESHOLD = 0.5

app = FastAPI(title="Mini RAG API", description="Chatbot RAG: retrieval two-stage + generation via Ollama")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[int]    
    scores: list[float]

@app.get("/health")
def health():
    return {"status": "ok", "documents": len(docs)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = embed_model.encode(req.question)
    scores = embs @ q
    cand = np.argsort(scores)[::-1][:10].tolist()
    if scores[cand[0]] < THRESHOLD:
        return AskResponse(
            answer="Maaf, jawaban tidak ada di dokumen saya.",
            sources=[],
            scores=[],
        )
    top_local, top_scores = rerank(req.question, [docs[i] for i in cand], top_n=3)
    top3 = [cand[i] for i in top_local]

    context = "\n\n".join(docs[i] for i in top3)
    prompt = f"""Jawablah pertanyaan berikut HANYA berdasarkan konteks di bawah ini.
                Jika jawabannya tidak ada di konteks, katakan "Tidak ada di dokumen saya."
                Gunakan Bahasa Indonesia.

            Konteks:
            {context}
            
            Pertanyaan: {req.question}
            Jawaban:"""
    r = requests.post("http://localhost:11434/api/generate", json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False}, timeout=120)
    answer = r.json()["response"]

    return AskResponse(
        answer=answer,
        sources=[i + 1 for i in top3],
        scores=[round(s, 3) for s in top_scores],
    )
