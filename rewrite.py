import os
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

def rewrite_query(question, model="qwen2.5:3b"):
    """Tulis ulang pertanyaan menjadi lebih deskriptif dan spesifik.

    Args:
        question: pertanyaan asli user
        model: model Ollama yang dipakai untuk rewriting

    Returns:
        pertanyaan hasil tulis ulang (satu kalimat)
    """
    prompt = f"""Tulis ulang pertanyaan berikut menjadi satu pertanyaan yang lebih deskriptif dan spesifik, tanpa mengubah maknanya. Hanya keluarkan pertanyaan hasil tulis ulang, tanpa penjelasan tambahan.

Pertanyaan: {question}
Pertanyaan baru:"""

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    rewritten = r.json()["response"].strip().split("\n")[0].strip()
    return rewritten