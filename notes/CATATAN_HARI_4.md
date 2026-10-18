# Catatan Belajar — Hari 4

**Tanggal:** Minggu, 18 Oktober 2026 (diambil cuti Jumat 16 Okt, dikerjakan 18 Okt)
**Tema:** Evaluasi — mengukur kualitas retrieval dengan angka
**Status:** ✅ Hari 4 selesai — **GATE LULUS 8/10, minggu validasi resmi selesai** 🎉

---

## Ringkasan Hari Ini (dalam 1 paragraf)

Hari ini kamu berhenti menebak dan mulai mengukur. Script `evaluate.py` menguji retrieval-mu dengan 10 soal parafrase dan menghitung metriknya: **top-3 accuracy 8/10 (80%)** — lulus gate minggu validasi (≥7/10). Dua soal miss memberi diagnosis presisi tentang kelemahan sistemmu, dan satu hasil tak terduga (soal LoRA yang parafrase justru lulus dengan skor tertinggi) memberi insight penting tentang cara kerja query. Minggu validasi selesai — RAG v0-mu bukan lagi "kayaknya jalan", tapi **terbukti jalan dengan angka**.

---

## 1. Apa Itu Evaluasi Retrieval?

**Evaluasi** = mengukur kualitas sistem dengan angka, bukan feeling.

Yang kita ukur hari ini adalah **retrieval eval**: *dari 10 soal, berapa yang dokumen benarnya muncul di 3 teratas?* Ini evaluasi yang deterministik (tidak butuh LLM sebagai juri) — cocok untuk tahap ini. Evaluasi generation (apakah jawabannya benar?) adalah topik fase berikutnya.

### Dua metrik

| Metrik | Arti | Hasilmu |
| --- | --- | --- |
| **Top-1 accuracy** | Dokumen benar ada di peringkat #1 | **8/10 (80%)** |
| **Top-3 accuracy** | Dokumen benar ada di 3 teratas | **8/10 (80%)** |

**Kenapa top-3 yang dipakai sebagai gate?** Karena di `rag_chat.py`, LLM membaca 3 dokumen. Selama dokumen yang benar ada di antara ketiganya, sistem punya bahan untuk menjawab dengan benar. Top-1 itu bagus, tapi top-3 yang menentukan sistemmu bisa kerja.

### Kenapa soalnya parafrase?

10 soal evaluasi sengaja ditulis ulang dengan kata berbeda. Contoh: bukan "Apa itu LoRA?" melainkan "Teknik fine-tuning apa yang hemat parameter?". Ini menguji retrieval **makna**, bukan cocok-string. Kalau soalnya copy-paste persis, evaluasinya tidak membuktikan apa-apa.

---

## 2. Hasil Lengkap

| # | Soal | Dokumen benar | Hasil | Top-3 (skor) |
| --- | --- | --- | --- | --- |
| 1 | Jelaskan cara kerja RAG | #1 Apa itu RAG? | ❌ MISS (peringkat #17) | #18 (0.554), #15 (0.506), #25 (0.435) |
| 2 | Fungsi chunking? | #4 Chunking | ✅ | #4 (0.607) di #1 |
| 3 | Kemiripan dua embedding? | #5 Cosine similarity | ❌ MISS (peringkat #10) | #2 (0.629), #10 (0.479), #21 (0.456) |
| 4 | Kenapa LLM mengarang? | #10 Halusinasi | ✅ | #10 (0.579) di #1 |
| 5 | Parameter deterministik? | #12 Temperature | ✅ | #12 (0.618) di #1 |
| 6 | Fine-tuning hemat parameter? | #14 LoRA | ✅ | #14 (0.660) di #1 |
| 7 | Gunanya reranking? | #15 Reranking | ✅ | #15 (0.618) di #1 |
| 8 | LLM tanpa internet? | #21 Ollama | ✅ | #21 (0.544) di #1 |
| 9 | Menolak topik terlarang? | #19 Guardrail | ✅ | #19 (0.832) di #1 |
| 10 | Update dokumen vs fine-tuning? | #25 RAG vs fine-tuning | ✅ | #25 (0.856) di #1 |

**Catatan pola:** 8 soal yang lulus semuanya menempatkan dokumen benar di **peringkat #1** — retrieval-mu bukan cuma "ketemu", tapi ketemu dengan tepat. Dua soal dengan skor tertinggi (0.832 dan 0.856) adalah soal yang kata-katanya paling mirip dengan dokumennya — petunjuk tentang cara kerja embedding.

---

## 3. Analisis 2 Miss (Diagnosis Presisi)

### Miss 1 — Soal 1: "Jelaskan cara kerja RAG" → dokumen #1 ada di peringkat #17

**Penyebab:** pertanyaan bersifat prosedural ("cara kerja"), dokumen #1 bersifat definisional ("Apa itu RAG?"). Embedding menangkap kedekatan topik prosedural → dokumen #18 (evaluasi RAG), #15 (reranking), #25 (RAG vs fine-tuning) yang semuanya membahas *proses* RAG. Ini **vocabulary/conceptual mismatch**: query dan dokumen bicara tentang hal yang berdekatan tapi bukan hal yang sama.

### Miss 2 — Soal 3: "Bagaimana cara mengukur kemiripan dua embedding?" → dokumen #5 ada di peringkat #10

**Penyebab:** kata "embedding" muncul dua kali di query, sehingga dokumen #2 ("Apa itu embedding?", 0.629) menang telak. Dokumen cosine similarity (#5) kalah karena query tidak mengandung kata "cosine" atau "similarity" sama sekali. Ini **keyword dominance**: satu kata yang berulang mendominasi skor.

### Obatnya (backlog fase RAG Depth)

Kedua pola ini adalah masalah klasik retrieval dengan solusi yang sudah mapan:

- **Query expansion / HyDE** — memperkaya query sebelum retrieval.
- **Reranking** — tahap kedua dengan model lebih akurat untuk mengurutkan ulang kandidat.
- **Chunking lebih baik** — dokumen yang terstruktur lebih mudah cocok.
- **Model embedding lebih besar** (mis. 768 dimensi) — menangkap nuansa lebih halus.

Inilah bedanya "tahu ada masalah" vs "tahu masalahnya apa": evaluasi memberimu **pola**, dan pola memberimu **obat yang tepat**.

---

## 4. Insight Tak Terduga: Paradoks LoRA

Ingat Hari 3? Pertanyaan langsung **"Apa itu LoRA?"** gagal total (dokumen LoRA tidak masuk top-3). Tapi hari ini soal parafrase **"Teknik fine-tuning apa yang hemat parameter?"** justru mendapat dokumen LoRA di **#1 dengan skor 0.660** — salah satu skor tertinggi!

**Pelajaran:** query yang **lebih deskriptif** memberi retrieval lebih banyak "petunjuk makna" untuk dipegang. "Apa itu LoRA?" hanya mengandalkan satu kata langka (LoRA) yang model kecil tidak kenal baik. Parafrase yang menyebut "fine-tuning" + "hemat parameter" memberi konteks yang cocok dengan isi dokumen jawabannya.

Implikasi praktisnya besar: di RAG production, **cara user bertanya sangat memengaruhi hasil** — dan teknik seperti query rewriting (menulis ulang query user menjadi lebih deskriptif sebelum retrieval) ada justru untuk mengatasi ini.

---

## 5. Pelajaran Terbesar Hari Ini

**Evaluasi mengubah feeling menjadi angka — dan angka mengubah "kayaknya ada masalah" menjadi "ini masalahnya, ini obatnya".**

Alur kerja AI Engineer yang benar, yang kamu praktikkan minggu ini:

1. **Bangun** (Hari 1–3) → 2. **Ukur** (Hari 4) → 3. **Diagnosis dari pola** (analisis miss) → 4. **Perbaiki dengan obat yang tepat** (fase berikutnya).

Kebalikannya — memperbaiki berdasarkan 2 anekdot tanpa mengukur — adalah premature optimization: berisiko memperbaiki bagian yang tidak rusak.

---

## 6. Gate Minggu Validasi: LULUS ✅

Tiga bukti gate, semuanya terpenuhi:

| Bukti | Status |
| --- | --- |
| **Kode jalan** — RAG v0 end-to-end (retrieval → threshold → generation via Ollama) | ✅ |
| **Evaluasi ≥7/10** — top-3 accuracy 8/10 | ✅ |
| **Explain-back** — embedding, retrieval, API Ollama, threshold dijelaskan dengan kata sendiri sepanjang minggu | ✅ |

**Minggu validasi resmi selesai** — setengah hari lebih cepat dari jadwal, dan sepenuhnya berbasis penguasaan.

---

## 7. Alur Kerja Penutup

```
git add . && git commit -m "day 4: evaluation 8/10" && git push
```

---

## 8. Berikutnya: Fase RAG Depth

Target fase berikutnya: **perbaiki 2 miss ini dan naikkan skor ke 9–10/10**, dengan senjata yang sekarang kamu pahami *kenapa* dibutuhkan: reranking, chunking yang lebih baik, dan query expansion. Setelah itu RAG-mu dibungkus menjadi API (FastAPI + Docker) — langkah pertama menuju proyek portofolio yang ter-deploy.

---
