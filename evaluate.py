import json
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/docs.json"
EMB_PATH = "data/embeddings.npy"
FAQ_PATH = "data/dataset_faq_rag_llm.jsonl"
EVAL_PATH = "data/dataset_eval_set.jsonl"

EXPECTED_DOC = [0, 3, 4, 9, 11, 13, 14, 20, 18, 24]

GATE = 7

def load_or_die(path, desc):
    try:
        return open(path, encoding="utf-8")
    except FileNotFoundError:
        print(f"File tidak ketemu: {path} ({desc})")
        print("Sesuaikan path di bagian konfigurasi atas script ini.")
        raise SystemExit(1)

docs = json.load(load_or_die(DOCS_PATH, "teks dokumen"))
embs = np.load(EMB_PATH)
faq = [json.loads(l)["question"] for l in load_or_die(FAQ_PATH, "daftar FAQ")]
eval_rows = [json.loads(l) for l in load_or_die(EVAL_PATH, "soal evaluasi")]

model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Menilai {len(eval_rows)} soal evaluasi...\n")

top1_hits = 0
top3_hits = 0

for i, row in enumerate(eval_rows):
    expected = EXPECTED_DOC[i]
    q = model.encode(row["question"])
    scores = embs @ q
    top3 = np.argsort(scores)[::-1][:3]

    hit_top1 = top3[0] == expected
    hit_top3 = expected in top3
    top1_hits += hit_top1
    top3_hits += hit_top3

    mark = "✅" if hit_top3 else "❌"
    print(f"{mark} Soal {i+1}: {row['question']}")
    print(f"   Dokumen yang benar : #{expected+1} {faq[expected]}")
    print(f"   Top-3 ter-retrieve : " +
          ", ".join(f"#{j+1} ({scores[j]:.3f})" for j in top3))
    if not hit_top3:
        print(f"   -> MISS. Dokumen benar ada di peringkat "
              f"#{int(np.argsort(scores)[::-1].tolist().index(expected))+1}")
    print()

n = len(eval_rows)
print("=" * 50)
print(f"Top-1 accuracy : {top1_hits}/{n} ({top1_hits/n:.0%})")
print(f"Top-3 accuracy : {top3_hits}/{n} ({top3_hits/n:.0%})")
print("=" * 50)
if top3_hits >= GATE:
    print(f"🎉 LULUS GATE ({top3_hits}/{n} >= {GATE}/{n}). Minggu validasi selesai!")
else:
    print(f"Belum lulus gate ({top3_hits}/{n} < {GATE}/{n}). "
          f"Lihat soal yang MISS di atas — di situlah yang perlu diperbaiki.")
