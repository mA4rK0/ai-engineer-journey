import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rerank import rerank
from rewrite import rewrite_query

DOCS_PATH = "data/docs.json"
EMB_PATH = "data/embeddings.npy"
FAQ_PATH = "data/dataset_faq_rag_llm.jsonl"
EVAL_PATH = "data/dataset_eval_set.jsonl"

EXPECTED_DOC = [0, 3, 4, 9, 11, 13, 14, 20, 18, 24]

CANDIDATES = 10
TOP_N = 3

def load_or_die(path, desc):
    try:
        return open(path, encoding="utf-8")
    except FileNotFoundError:
        print(f"File tidak ketemu: {path} ({desc})")
        raise SystemExit(1)

docs = json.load(load_or_die(DOCS_PATH, "teks dokumen"))
embs = np.load(EMB_PATH)
faq = [json.loads(l)["question"] for l in load_or_die(FAQ_PATH, "daftar FAQ")]
eval_rows = [json.loads(l) for l in load_or_die(EVAL_PATH, "soal evaluasi")]

model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Menilai {len(eval_rows)} soal dengan three-stage retrieval "
      f"(rewrite -> bi-encoder top-{CANDIDATES} -> cross-encoder top-{TOP_N})...\n")

top1_hits = 0
top3_hits = 0

for i, row in enumerate(eval_rows):
    expected = EXPECTED_DOC[i]
    rewritten = rewrite_query(row["question"])

    q = model.encode(rewritten)
    scores = embs @ q
    cand_idx = np.argsort(scores)[::-1][:CANDIDATES].tolist()

    cand_docs = [docs[j] for j in cand_idx]
    top_local, top_scores = rerank(rewritten, cand_docs, top_n=TOP_N)
    top3 = [cand_idx[j] for j in top_local]

    hit_top1 = top3[0] == expected
    hit_top3 = expected in top3
    top1_hits += hit_top1
    top3_hits += hit_top3

    mark = "✅" if hit_top3 else "❌"
    print(f"{mark} Soal {i+1}: {row['question']}")
    print(f"   Ditulis ulang -> {rewritten}")
    print(f"   Dokumen yang benar : #{expected+1} {faq[expected]}")
    print(f"   Top-3 setelah rerank: " +
          ", ".join(f"#{j+1} ({s:.3f})" for j, s in zip(top3, top_scores)))
    if not hit_top3:
        print(f"   -> MISS.")
    print()

n = len(eval_rows)
print("=" * 50)
print(f"BASELINE bi-encoder saja         : 8/10 (80%)")
print(f"BASELINE bi-encoder + rerank     : 9/10 (90%)")
print(f"Top-1 accuracy (dengan rewrite)  : {top1_hits}/{n} ({top1_hits/n:.0%})")
print(f"Top-3 accuracy (dengan rewrite)  : {top3_hits}/{n} ({top3_hits/n:.0%})")
print("=" * 50)
if top3_hits == 10:
    print("🎉 10/10! Kasus LoRA dan soal 1 tertutup — RAG Depth selesai sempurna.")
elif top3_hits > 9:
    print(f"Naik ke {top3_hits}/10 — rewriting membantu.")
elif top3_hits == 9:
    print("Tetap 9/10. Rewriting tidak mengubah hasil — lihat soal yang MISS.")
else:
    print(f"Turun ke {top3_hits}/10 — rewriting justru merusak. Diagnosis diperlukan.")
