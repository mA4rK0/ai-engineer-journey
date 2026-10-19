# Catatan Belajar — Hari 5

**Tanggal:** Senin, 19 Oktober 2026
**Tema:** Reranking — two-stage retrieval (Hari 1 fase RAG Depth)
**Status:** ✅ Hari 5 selesai — **skor naik 8/10 → 9/10, GATE RAG DEPTH LULUS** 🎉

---

## Ringkasan Hari Ini (dalam 1 paragraf)

Hari ini fase RAG Depth dimulai dengan misi "buktikan RAG bisa bagus" (gate ≥9/10). Senjatanya: **two-stage retrieval** — bi-encoder yang cepat menyaring 25 dokumen menjadi 10 kandidat, lalu cross-encoder yang teliti mengurutkan ulang 10 kandidat menjadi 3 terbaik. Hasilnya: **top-3 accuracy naik dari 8/10 menjadi 9/10** — gate lulus. Satu miss terperbaiki (soal 3), satu miss bertahan (soal 1) dengan diagnosis yang justru lebih berharga: reranking tidak bisa memperbaiki recall. Tiga pelajaran hari ini adalah pelajaran production-grade yang dipakai semua sistem RAG sungguhan.

---

## 1. Konsep: Two-Stage Retrieval

### Masalahnya

Retrieval-mu sejauh ini (bi-encoder) itu **cepat tapi dangkal**: dia mengubah query dan dokumen menjadi vektor *terpisah*, lalu membandingkan angkanya. Cepat — 25 dokumen disaring sekejap — tapi dia tidak pernah benar-benar "membaca" apakah dokumen itu menjawab pertanyaan.

### Dua jenis model

|  | Bi-encoder | Cross-encoder |
| --- | --- | --- |
| Cara kerja | Query dan dokumen di-encode **terpisah** jadi vektor, lalu dibandingkan | Query + dokumen dibaca **bersamaan** dalam satu model |
| Kecepatan | Sangat cepat | Lambat (harus baca tiap pasangan) |
| Ketelitian | Dangkal | Teliti |
| Analogi | **Satpam yang melirik dari jauh** | **Pewawancara yang membaca CV baris per baris** |
| Peran | Tahap 1: saring banyak → sedikit (recall) | Tahap 2: pilih yang terbaik (precision) |

### Solusinya: gabungkan keduanya

1. **Tahap 1 (bi-encoder, cepat):** saring 25 dokumen → 10 kandidat.
2. **Tahap 2 (cross-encoder, teliti):** urutkan ulang 10 kandidat → 3 terbaik.

Ini pola standar semua RAG production: tahap cepat untuk recall, tahap teliti untuk precision. Cross-encoder tidak mungkin dipakai untuk ribuan dokumen sekaligus (terlalu lambat) — makanya dia hanya bekerja pada kandidat yang sudah disaring.

### Model yang dipakai

`cross-encoder/ms-marco-MiniLM-L-6-v2` (\~90MB, download sekali):

- **MS MARCO** = dataset raksasa dari Microsoft berisi jutaan pasangan (query, dokumen) dengan label relevan/tidak — standar emas untuk melatih model reranking.
- **MiniLM-L-6** = arsitektur kecil-cepat, sekeluarga dengan model embedding-mu (all-MiniLM-L6-v2).

---

## 2. Bedah Kode `rerank.py`

```python
from sentence_transformers import CrossEncoder

def rerank(query, docs, top_n=3):
    model = get_model()                          # load sekali, pakai berkali-kali
    pairs = [(query, d) for d in docs]            # bentuk pasangan (query, dokumen)
    scores = model.predict(pairs)                # skor relevansi per pasangan
    ranked = sorted(range(len(docs)),
                    key=lambda i: scores[i],
                    reverse=True)                # urutkan indeks dari skor tertinggi
    return ranked[:top_n], [...]                 # ambil top_n teratas
```

Tiga baris intinya:

1. **`pairs = [(query, d) for d in docs]`** — inilah bedanya dengan bi-encoder: query tidak diubah jadi vektor sendirian, melainkan **dipasangkan** dengan tiap dokumen dan dibaca bersamaan oleh model.
2. **`scores = model.predict(pairs)`** — model mengeluarkan skor relevansi untuk tiap pasangan.
3. **`sorted(..., reverse=True)[:top_n]`** — urutkan dari skor tertinggi, ambil yang teratas. (Pola yang sama dengan `argsort` di Hari 2, versi Python murni.)

### Catatan Windows: warning symlink

Saat download model, muncul warning tentang symlink di Windows. **Tidak berbahaya** — cache tetap bekerja, hanya memakai disk sedikit lebih banyak. Bisa diabaikan. (Kalau mau hilang: aktifkan Developer Mode di Windows.)

---

## 3. Bedah Alur `evaluate_rerank.py`

Alur per soal (10 soal yang sama dengan evaluasi Hari 4):

```
Soal → TAHAP 1: bi-encoder ambil 10 kandidat (embs @ q, seperti biasa)
     → TAHAP 2: cross-encoder urutkan ulang 10 kandidat → 3 terbaik
     → Skor: apakah dokumen yang benar ada di 3 teratas?
```

Dua konstanta yang mengatur perilakunya:

- `CANDIDATES = 10` — berapa kandidat yang disaring tahap 1. Lebih besar = recall lebih aman, tapi rerank lebih lambat.
- `TOP_N = 3` — berapa yang diambil tahap 2 (menyesuaikan `rag_chat.py` yang memberi LLM 3 dokumen).

---

## 4. Hasil: 8/10 → 9/10

| Metrik | Baseline (Hari 4) | Dengan rerank (Hari 5) |
| --- | --- | --- |
| Top-1 accuracy | 8/10 (80%) | 6/10 (60%) ⬇ |
| **Top-3 accuracy** | **8/10 (80%)** | **9/10 (90%)** ⬆ |

### Yang berubah per soal

- ✅ **Soal 3 (cosine similarity) TERPERBAIKI** — kemarin MISS (peringkat #10), sekarang dokumen benar masuk top-3. Reranker yang teliti berhasil mengangkat dokumen yang tepat.
- ❌ **Soal 1 masih MISS** — dengan diagnosis baru (lihat Pelajaran 1).
- 8 soal lainnya tetap lulus.

### Top-1 turun, top-3 naik — tidak apa-apa

Top-1 turun 8/10 → 6/10 karena cross-encoder lebih "berpendapat" di peringkat pertama (soal 4 dan 8: dia menaruh dokumen lain di #1). Tapi **top-3 yang dipakai RAG** — LLM-mu membaca 3 dokumen, bukan 1. Metrik yang benar untuk sistemmu adalah top-3, dan itu naik.

---

## 5. Tiga Pelajaran Production-Grade

### Pelajaran 1: Reranking tidak bisa memperbaiki recall ⚠️ (paling penting)

Soal 1 masih MISS — dan penyebabnya adalah pelajaran arsitektur: dokumen yang benar **tidak masuk 10 kandidat** tahap 1 (`tidak masuk kandidat`). Reranker tidak pernah melihatnya.

**Prinsipnya:** two-stage retrieval selalu dibatasi oleh **recall tahap 1**. Pewawancara seteliti apa pun tidak bisa memilih kandidat yang tidak lolos satpam. Obatnya bukan reranker yang lebih bagus, melainkan:

- Kandidat lebih banyak (top-20, bukan top-10), atau
- Tahap 1 yang lebih baik (query rewriting — menulis ulang query menjadi lebih deskriptif sebelum retrieval).

### Pelajaran 2: Pilih metrik yang sesuai dengan sistemmu

Top-1 dan top-3 bisa bergerak berlawanan arah (yang satu turun, yang satu naik). **Metrik yang benar = metrik yang mencerminkan cara sistemmu dipakai.** Karena `rag_chat.py` memberi LLM 3 dokumen, top-3 accuracy adalah metrik kebenaranmu — bukan top-1.

Pelajaran umum: jangan mengoptimasi metrik yang salah. Sistem yang "terlihat lebih buruk" di metrik yang salah bisa jadi justru lebih baik di metrik yang benar.

### Pelajaran 3: Jangan bandingkan skor antar tahap

Skor cross-encoder bisa **negatif** (-3.321, -6.018) — itu **raw logits** (skor mentah model), bukan cosine similarity (yang rentangnya -1 sampai 1). Keduanya hanya bermakna untuk *ranking di dalam tahapnya masing-masing*.

Analogi: membandingkan "0.554 vs -3.321" itu seperti membandingkan suhu Celsius dengan Fahrenheit tanpa konversi — angkanya beda skala, yang penting urutannya.

---

## 6. Gate Fase RAG Depth: LULUS ✅

| Bukti | Status |
| --- | --- |
| Skor naik terukur: 8/10 → 9/10 | ✅ |
| Memahami two-stage retrieval dan bisa menjelaskan trade-off-nya | ✅ |
| Mendiagnosis miss yang bertahan (recall bottleneck) dengan tepat | ✅ |

**Satu miss tersisa (soal 1)** menjadi backlog yang jelas: recall tahap 1. Obatnya (query rewriting / kandidat lebih banyak) sudah teridentifikasi — bukan misteri lagi.

---

## 7. Alur Kerja Penutup

```
git add . && git commit -m "day 5: reranking 9/10" && git push
```

---