# Catatan Belajar — Hari 2

**Tanggal:** Rabu, 14 Oktober 2026
**Tema:** Retrieval — mencari dokumen paling mirip dengan pertanyaan
**Status:** ✅ Hari 2 selesai (retrieval CLI jalan + 3 jenis pertanyaan diuji + commit)

---

## Ringkasan Hari Ini (dalam 1 paragraf)

Hari ini kamu membangun **setengah kedua dari fondasi RAG: retrieval**. Kemarin (Hari 1) kamu mengubah 25 dokumen menjadi vektor dan menyimpannya. Hari ini kamu memakai vektor-vektor itu untuk menjawab pertanyaan: pertanyaan diubah menjadi vektor juga, lalu dibandingkan dengan semua vektor dokumen untuk menemukan 3 dokumen yang maknanya paling dekat. Kamu mengujinya dengan 3 jenis pertanyaan (parafrase, persis, dan ngawur), dan ketiganya berperilaku persis seperti yang seharusnya. Kamu juga belajar satu koreksi penting: ambang skor tidak boleh ditebak — harus dikalibrasi dari data nyata.

---

## 1. Apa Itu Retrieval?

### Definisi

**Retrieval** = mencari dokumen yang paling relevan dengan pertanyaan, berdasarkan **makna** — bukan berdasarkan kata yang sama persis.

### Analogi

Bayangkan perpustakaan dengan 25 buku. Kemarin kamu sudah memberi setiap buku "koordinat makna" (embedding). Hari ini, ketika ada pertanyaan masuk, kamu memberi pertanyaan itu koordinat juga, lalu mencari 3 buku yang koordinatnya paling dekat dengan koordinat pertanyaan. Buku yang koordinatnya dekat = buku yang maknanya mirip.

### Posisi retrieval dalam RAG

RAG punya dua sisi:

- **Retrieval** (Hari 1–2, sudah kamu bangun): **mencari** dokumen yang relevan.
- **Generation** (Hari 3–4, besok): **membuat** jawaban dari dokumen yang ditemukan, memakai LLM (Ollama).

Retrieval saja = "mesin pencari makna". Retrieval + generation = chatbot RAG yang utuh.

---

## 2. Cara Kerja Retrieval (4 Langkah)

Script `retrieve.py` melakukan 4 langkah:

### Langkah 1 — Load pasangan file kemarin

```python
docs = json.load(open("data/docs.json"))
embs = np.load("data/embeddings.npy")
```

- `embs[i]` = vektor dokumen ke-i.
- `docs[i]` = teks dokumen ke-i.
- Urutannya harus persis sama (yang sudah dijelaskan di catatan Hari 1).

### Langkah 2 — Embedding pertanyaan (WAJIB model yang sama!)

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
q = model.encode(query)
```

Pertanyaan diubah menjadi vektor 384 angka dengan **model yang sama persis** dengan yang dipakai untuk dokumen. Aturan keras dari Hari 1: vektor dari model berbeda tidak bisa dibandingkan.

### Langkah 3 — Hitung cosine similarity ke SEMUA dokumen sekaligus

```python
scores = embs @ q
```

### Langkah 4 — Ambil 3 skor tertinggi

```python
top3 = np.argsort(scores)[::-1][:3]
```

- `np.argsort(scores)` = mengurutkan indeks dari skor **terkecil ke terbesar**.
- `[::-1]` = membalik urutannya, sehingga menjadi **terbesar ke terkecil**.
- `[:3]` = ambil 3 teratas.
- Hasilnya: 3 indeks dokumen dengan skor tertinggi, lalu teksnya diambil dari `docs.json`.

---

## 3. Lambang `@` = Perkalian Matriks

`@` adalah operator khusus Python untuk **perkalian matriks** (resmi sejak Python 3.5). Berbeda dengan `*` yang hanya mengalikan per elemen.

```
embs  = matriks (25, 384)   → 25 dokumen, tiap dokumen 384 angka
q     = vektor  (384,)      → pertanyaanmu
embs @ q  =  (25,)          → 25 skor, satu per dokumen
```

Yang terjadi di balik layar: setiap baris `embs` di-dot-product dengan `q`.

- Baris ke-0 × `q` = skor dokumen ke-0.
- Baris ke-1 × `q` = skor dokumen ke-1.
- Dan seterusnya sampai baris ke-24.

Hasilnya: 25 skor dalam **satu baris kode**, dalam sekali jalan.

### Kenapa dot product = cosine similarity?

Karena semua vektor sudah **ternormalisasi** (L2 norm = 1, berkat tahap Normalize di Hari 1), maka dot product dan cosine similarity menghasilkan angka yang sama. Inilah manfaat praktis dari normalisasi: perhitungan kemiripan menjadi semurah perkalian biasa.

---

## 4. Tiga Eksperimen Hari Ini (Bukti Retrieval Jalan)

### Eksperimen 1 — Pertanyaan parafrase: "Jelaskan cara kerja RAG"

| Peringkat | Skor  | Dokumen                                        |
| --------- | ----- | ---------------------------------------------- |
| #1        | 0.517 | Apa yang diukur dalam evaluasi RAG?            |
| #2        | 0.509 | Apa itu reranking dalam RAG?                   |
| #3        | 0.394 | Kenapa RAG lebih praktis daripada fine-tuning? |

**Kesimpulan:** Ketiga dokumen semuanya bertopik RAG. Retrieval-mu mencari **makna**, bukan kata — terbukti, karena pertanyaanmu tidak mengandung kata "evaluasi" atau "reranking" sama sekali, tetapi dokumen-dokumen itu tetap muncul.

**Nuansa jujur:** Dokumen "Apa itu RAG?" tidak masuk top-3. Kenapa? Karena pertanyaanmu bersifat prosedural ("cara kerja"), dan embedding model kecil ini menangkap kedekatan topik, bukan kecocokan persis. Ini **normal** untuk dataset 25 dokumen + model 384 dimensi. Sistemmu tidak rusak; ia hanya belum canggih. Inilah alasan kenapa fase berikutnya punya reranking, chunking lebih baik, dan evaluasi.

### Eksperimen 2 — Pertanyaan persis: "Apa itu quantization?"

| Peringkat | Skor      | Dokumen                                        |
| --------- | --------- | ---------------------------------------------- |
| #1        | **0.648** | **Apa itu quantization? (dokumen yang benar)** |
| #2        | 0.581     | Apa itu context window?                        |
| #3        | 0.541     | Apa itu prompt engineering?                    |

**Kesimpulan:** Kemenangan bersih — pertanyaan persis → dokumen yang benar di peringkat #1 dengan skor tertinggi yang jelas. Kalau semua query seperti ini, RAG sudah sempurna.

### Eksperimen 3 — Pertanyaan ngawur: "Berapa harga cabai hari ini?"

| Peringkat | Skor  | Dokumen                      |
| --------- | ----- | ---------------------------- |
| #1        | 0.426 | Apa itu AI agent?            |
| #2        | 0.418 | Apa itu halusinasi pada LLM? |
| #3        | 0.390 | Apa itu Ollama?              |

**Kesimpulan:** Semua skor **rendah** dan dokumennya tidak nyambung. Sistem mengembalikan "yang paling tidak buruk", bukan yang benar. Ini perilaku yang **diharapkan** — dan ini eksperimen yang paling penting hari ini, karena mengajarkan pelajaran tentang threshold (lihat Bagian 5).

---

## 5. Koreksi Penting: Ambang Skor Harus Dikalibrasi, Bukan Ditebak

### Kesalahan di `retrieve.py`

Di script `retrieve.py`, tertulis panduan: "0.0–0.3 = tidak nyambung". **Itu salah untuk setup ini.** Buktinya: pertanyaan cabai yang jelas-jelas tidak nyambung saja mendapat skor 0.42 — jauh di atas 0.3.

### Ambang yang terkalibrasi (dari data nyata hari ini)

| Skor          | Arti (untuk model + dataset ini)                                    |
| ------------- | ------------------------------------------------------------------- |
| **> 0.6**     | **Kuat** — dokumen sangat relevan (contoh: quantization, 0.648)     |
| **0.5 – 0.6** | **Sedang** — dokumen relevan (contoh: parafrase RAG, 0.51–0.52)     |
| **< 0.5**     | **Lemah** — dokumen tidak cukup nyambung (contoh: cabai, 0.39–0.43) |

### Pelajaran umumnya

**Threshold (ambang batas) jangan ditebak — harus dikalibrasi dari data nyata**, persis seperti yang kamu lakukan hari ini dengan 3 eksperimen. Setiap kombinasi model + dataset punya "skala skor" sendiri. Model lain atau dataset lain akan punya angka yang berbeda.

---

## 6. Threshold → Cikal Bakal Guardrail

Eksperimen cabai mengajarkan satu konsep production yang penting:

> **Di RAG sungguhan, kamu memasang threshold.** Jika skor dokumen terbaik masih di bawah ambang (misalnya < 0.5), bot **tidak boleh** menjawab dari dokumen yang salah. Sebaliknya, bot menjawab: _"Maaf, jawaban tidak ada di dokumen saya."_

Ini adalah bentuk paling sederhana dari **guardrail** — aturan yang membatasi perilaku chatbot agar tidak mengarang jawaban dari dokumen yang tidak relevan. Tanpa threshold, RAG akan selalu "menjawab" — bahkan ketika ia seharusnya bilang tidak tahu. Nanti di fase LLM depth, guardrail menjadi topik tersendiri.

---

---

## 8. Alur Kerja & Progres

### Perintah penutup Hari 2

```
git add . && git commit -m "day 2: retrieval CLI" && git push
```

### Status perjalanan

- **Streak:** 2 hari (Hari 1 ✅, Hari 2 ✅).
- **Kecepatan:** kamu sudah **setengah hari lebih cepat** dari jadwal roadmap — bukti bahwa sistem mastery-based bekerja: kamu tidak ditahan kalender, kamu maju secepat penguasaanmu.
- **Gate Minggu 1–2** (pengingat): RAG v0 jalan (retrieval + jawaban Ollama), evaluasi ≥7/10, bisa menjelaskan embedding & RAG sendiri.

---

## 9. Besok: Hari 3 — Generation (RAG v0 yang Bisa Menjawab)

Tugas besok: **menyambungkan Ollama** ke retrieval yang sudah jalan. Alurnya akan menjadi:

```
Pertanyaan → Retrieval (top-3 dokumen) → Dokumen + pertanyaan dikirim ke Ollama
→ Ollama menulis jawaban berdasarkan dokumen → Jawaban ditampilkan
```

Untuk pertama kalinya, sistemmu tidak hanya _mencari_ — ia _menjawab_. Itulah saat RAG mini-mu berubah dari "mesin pencari" menjadi "chatbot".

---
