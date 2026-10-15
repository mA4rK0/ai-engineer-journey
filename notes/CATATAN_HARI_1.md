# Catatan Belajar — Hari 1

**Tanggal:** Selasa, 13 Oktober 2026
**Tema:** Embedding pertama — dari kalimat menjadi vektor
**Status:** ✅ Hari 1 selesai (setup + dataset + embedding pertama + commit)

---

## Ringkasan Hari Ini (dalam 1 paragraf)

Hari ini kamu belajar fondasi pertama RAG: **embedding**. Kamu belajar bahwa sebuah kalimat bisa diubah menjadi daftar 384 angka yang disebut vektor, bahwa vektor itu dibuat oleh model `all-MiniLM-L6-v2` melalui pipa 3 tahap (Transformer → Pooling → Normalize), dan bahwa vektor-vektor itu disimpan ke dalam dua file berpasangan (`embeddings.npy` + `docs.json`) yang besok akan dipakai untuk mencari dokumen paling relevan. Semua materi hari ini adalah fondasi langsung menuju chatbot RAG-mu.

---

## 1. Apa Itu Embedding?

### Definisi

**Embedding** adalah cara mengubah teks menjadi daftar angka (vektor) yang menangkap **makna** teks tersebut. Komputer tidak mengerti kata — komputer hanya mengerti angka. Jadi sebelum komputer bisa "mencari makna", teks harus diterjemahkan dulu menjadi angka.

### Analogi anak SD: "rasa" makanan

Bayangkan ada **384 macam rasa**: manis, asam, asin, pahit, gurih, dan seterusnya sampai 384.

Setiap makanan kamu cicipi dan beri nilai:

- **Es krim coklat** → manis 9, pahit 2, asam 0, ...
- **Kopi hitam** → pahit 9, manis 1, asam 3, ...

Kalau ada makanan baru yang nilai rasanya mirip es krim (manis 8, pahit 3), kamu langsung tahu: "ini pasti manis seperti es krim!" — tanpa perlu mencicipinya.

Kalimat bekerja dengan cara yang sama. Setiap kalimat diberi **384 nilai "rasa makna"**:

- Kalimat yang artinya mirip → nilai-nilainya mirip.
- Kalimat yang tidak nyambung → nilai-nilainya beda jauh.

Itulah embedding: **"rasa" sebuah kalimat, dinyatakan dalam 384 angka.**

### Istilah penting

- **Vektor** = daftar angka. Contoh: `[-0.08, 0.08, -0.01, ...]` (total 384 angka).
- **Dimensi** = jumlah angka dalam vektor. Vektor embedding kita berdimensi **384**.
- **Distributed representation** = makna sebuah kalimat tersebar di _seluruh_ pola 384 angka, bukan di satu angka tertentu. Tidak ada satu dimensi yang artinya "RAG". Kalau satu angka diganti sedikit, maknanya hampir tidak berubah.

---

## 2. Isi Perut SentenceTransformer (Pipa 3 Tahap)

Ketika kamu menjalankan `print(model)`, yang tampil adalah arsitektur model embedding — sebuah **pipa 3 tahap**:

```
Teks masuk → (0) Transformer → (1) Pooling → (2) Normalize → Vektor 384 angka keluar
```

### Tahap 0 — Transformer (`BertModel`)

- Teks dibaca, lalu **setiap token** (potongan kata) diubah menjadi vektornya masing-masing.
- Kalau kalimatmu punya 10 token, keluar 10 vektor.
- Output tahap ini disebut **`token_embeddings`** (embedding per token).
- Tugas tahap ini: memahami konteks tiap kata di dalam kalimat.

### Tahap 1 — Pooling (`mean`)

- 10 vektor token itu **dirata-rata** (`mean` = rata-rata) menjadi **satu** vektor 384 angka untuk seluruh kalimat.
- Inilah "embedding kalimat"-nya.
- Analogi: 10 pendapat anggota tim dirata-rata menjadi 1 keputusan tim.

### Tahap 2 — Normalize

- Vektor hasil rata-rata di-**skala** supaya **panjangnya tepat 1** (disebut L2 norm = 1).
- Kenapa? Ini trik matematis: kalau semua vektor panjangnya 1, maka **cosine similarity = sekadar dot product biasa** — perhitungannya jadi jauh lebih murah dan cepat.
- Bukti tahap ini bekerja: vektor "Apa itu RAG?" yang kita hitung punya **L2 norm tepat 1.0**.

### Kalimat kesimpulan

Teks masuk → tiap token jadi vektor → dirata-rata jadi 1 vektor 384 angka → dinormalisasi. Itulah embedding.

---

## 3. Kode Embedding Pertama (Bedah Baris per Baris)

```python
from sentence_transformers import SentenceTransformer
import json
import numpy as np

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
```

| Baris                                     | Artinya                                                                                                                                                                                                                                               |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SentenceTransformer("all-MiniLM-L6-v2")` | Load model embedding. `all-MiniLM-L6-v2` = model kecil (~90MB), cepat, output 384 dimensi, cocok untuk belajar dan laptop tanpa GPU. **Pertama kali** dijalankan, model akan di-download (sekali saja). Selanjutnya langsung memakai cache (offline). |
| `model.encode(teks)`                      | Mengubah teks menjadi vektor embedding. **Deterministik**: teks yang sama selalu menghasilkan vektor yang sama persis.                                                                                                                                |
| `vec.shape` → `(384,)`                    | Bentuk vektor: berisi 384 angka.                                                                                                                                                                                                                      |
| `vec[:5]`                                 | **Slicing**: tampilkan 5 angka pertama saja, supaya terminal tidak kebanjiran angka. Vektornya tetap 384 angka — 5 itu hanya intipan. Ibarat buku 384 halaman, kamu hanya membuka 5 halaman pertama.                                                  |
| `vec.dtype` → `float32`                   | Tipe data tiap angka: bilangan desimal 32-bit.                                                                                                                                                                                                        |

### Membaca output terminal

```
Warning: You are sending unauthenticated requests to the HF Hub...
```

**Bukan error.** Ini hanya Hugging Face memberi tahu bahwa kamu download tanpa login, jadi kecepatan download dibatasi. Untuk belajar tidak masalah sama sekali. Kalau download terasa lambat, baru perlu membuat token gratis di huggingface.co.

```
Loading weights: 100% | 103/103
```

**103** adalah jumlah "tensor bobot" di dalam model. 100% artinya semua bobot sudah masuk memori dan model siap dipakai.

```
SentenceTransformer(
  (0): Transformer(...)   # tahap Transformer
  (1): Pooling(...)       # tahap Pooling (mean)
  (2): Normalize(...)     # tahap Normalize
)
```

Ini adalah hasil `print(model)` — arsitektur pipa 3 tahap yang dibahas di Bagian 2.

---

## 4. Bentuk Vektor: 384 Angka "Apa itu RAG?"

Vektor asli kalimat **"Apa itu RAG?"** (dihitung dengan model yang sama persis):

```
Kalimat       : Apa itu RAG?
Dimensi       : (384,)
5 angka pertama: [-0.08592518,  0.0858885,  -0.01514577, -0.06547566, -0.1477347]
Tipe data     : float32
```

### Statistik vektornya

| Ukuran               | Nilai   | Artinya                                                    |
| -------------------- | ------- | ---------------------------------------------------------- |
| Minimum              | -0.1594 | Angka terkecil di dalam vektor                             |
| Maksimum             | 0.1374  | Angka terbesar di dalam vektor                             |
| Mean (rata-rata)     | 0.0003  | Nilai-nilainya mengumpul di sekitar nol                    |
| Std (simpangan baku) | 0.051   | Sebaran nilainya kecil — tipikal untuk embedding model ini |
| **L2 norm**          | **1.0** | **Bukti tahap Normalize bekerja**                          |

### Cara membaca visualisasi vektor (heatmap)

- Setiap kotak = 1 angka dari 384 angka.
- **Oranye** = angka positif, **biru** = angka negatif.
- Makin pekat warnanya = angkanya makin besar (positif besar atau negatif besar).
- 5 kotak berbingkai = 5 angka pertama yang di-print.
- Kesimpulan visual: nilai-nilainya kecil dan mengumpul di nol — itu normal, bukan bug.

---

## 5. Dimensi: Kenapa 384?

**384 itu khusus model ini** (`all-MiniLM-L6-v2`). Model lain punya dimensi lain:

| Model                              | Dimensi |
| ---------------------------------- | ------- |
| all-MiniLM-L6-v2 (yang kamu pakai) | 384     |
| all-mpnet-base-v2                  | 768     |
| OpenAI text-embedding-3-small      | 1536    |
| OpenAI text-embedding-3-large      | 3072    |

**Trade-off:** dimensi lebih besar = kapasitas menangkap nuansa makna lebih besar, tetapi file embedding lebih besar dan pencarian lebih lambat.

### Aturan keras (wajib diingat)

**Query dan dokumen harus memakai model yang sama.** Vektor 384 dimensi tidak bisa dibandingkan dengan vektor 768 dimensi — ibarat membandingkan koordinat peta 2D dengan koordinat ruang 3D, tidak nyambung.

Untuk belajar, 384 sudah lebih dari cukup. Nanti kalau butuh kualitas retrieval lebih tinggi, baru naik ke 768.

---

## 6. File Hasil: `embeddings.npy` + `docs.json`

Dua file ini **sepasang** — yang satu tanpa yang lain tidak berguna.

### `embeddings.npy` — koordinat maknanya

- File biner format native NumPy yang menyimpan **matriks embedding: 25 baris × 384 kolom** (25 dokumen, tiap dokumen 1 vektor 384 angka).
- Tujuannya: **tidak mengulang kerjaan mahal**. Menghitung embedding itu berat (model neural network berjalan 25 kali). Kalau setiap kali retrieval kamu hitung ulang, laptop kerja keras untuk hasil yang sama persis — karena embedding itu deterministik. Jadi: hitung sekali, simpan, load berkali-kali.
- `.npy` dipilih karena format native NumPy: load-nya cepat dan bentuk matriks + `float32`-nya tetap utuh.

### `docs.json` — teks aslinya

- Daftar 25 teks dokumen, **urutannya persis sama** dengan baris di `.npy`:
  ```
  docs.json[3]  ↔  embeddings.npy baris ke-3
  docs.json[7]  ↔  embeddings.npy baris ke-7
  ```
- Tanpa file ini, kamu punya 25 vektor tetapi tidak tahu vektor itu mewakili teks apa — seperti punya 25 sidik jari tanpa nama pemiliknya.

### Cara keduanya bekerja besok (retrieval)

1. Pertanyaan user → diubah jadi embedding query.
2. Cari baris `.npy` yang paling dekat (cosine similarity).
3. Ketemu baris ke-7 → ambil `docs.json[7]` → tampilkan teksnya sebagai konteks jawaban.
4. Satu kalimat: `.npy` = maknanya dalam angka, `.json` = teksnya. Di proyek sungguhan nanti, peran dua file ini digantikan oleh **vector database**.

---

## 7. Model Tersimpan di Mana? (HF Cache)

Model yang di-download tersimpan otomatis di:

```
~/.cache/huggingface/hub/
├── blobs/                                              ← file bobot asli (di-share)
└── models--sentence-transformers--all-MiniLM-L6-v2    ← snapshot model (isinya symlink ke blobs)
```

Di Windows: `C:\Users\<namamu>\.cache\huggingface\hub`.

Tiga hal yang perlu diketahui:

1. **Otomatis, tidak perlu disentuh.** Pertama kali `SentenceTransformer("nama-model")` dipanggil, model di-download ke sana. Pemanggilan berikutnya langsung memakai cache (offline).
2. **Satu model = download sekali.** Dipakai 10 script berbeda pun tetap satu copy.
3. **Bisa dipindah/dihapus.** Mau ganti lokasi: set environment variable `HF_HUB_CACHE`. Mau download ulang dari nol: hapus foldernya.

Prinsip umum: semua library Hugging Face (transformers, sentence-transformers, diffusers) memakai cache yang sama.

---

## 8. Ollama & Model `qwen2.5:3b` (untuk Tahap Menjawab)

RAG punya dua sisi: **retrieval** (mencari dokumen, yang kamu pelajari hari ini) dan **generation** (menjawab pakai LLM). Untuk sisi menjawab, kita memakai **Ollama** — tool untuk menjalankan LLM open-weight secara lokal di laptop sendiri.

### Cara install & download model

1. Install Ollama dari ollama.com (download installer sesuai OS, install seperti aplikasi biasa — dia jalan otomatis sebagai background service).
2. Buka terminal, jalankan:
   ```
   ollama pull qwen2.5:3b
   ```
3. Tes dengan: `ollama run qwen2.5:3b` (langsung bisa chat di terminal; ketik `/bye` untuk keluar).

### Fakta tentang model ini

| Pertanyaan       | Jawaban                                                                                                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gratis permanen? | **Ya.** Qwen2.5 berlisensi Apache 2.0 — gratis selamanya, boleh dipakai komersial. Karena jalan 100% lokal, tidak ada biaya API apa pun. Sekali download, bisa offline selamanya. |
| Storage?         | **~1,9–2 GB.** Kecil, muat santai di laptop mana pun.                                                                                                                             |
| RAM saat jalan?  | ~3–4 GB. Laptop 8 GB ke atas aman.                                                                                                                                                |

### Cara memanggilnya dari Python (nanti, Hari 4)

Ollama menyediakan API lokal gratis di `http://localhost:11434`. Script RAG-mu akan mengirim prompt + konteks dokumen ke alamat itu, dan Ollama mengembalikan jawabannya.

---

## 9. Keputusan Tooling Hari Ini

### `pip` atau PDM? → **pip + venv**

- Semua dokumentasi (Hugging Face, PyTorch, LangChain, tutorial RAG) memakai pip — zero friction, copy command langsung jalan.
- `pip` / `venv` / `requirements.txt` adalah bahasa standar production & Docker — skill wajib calon AI Engineer.
- PDM bagus (lockfile rapi, tidak perlu activate manual), tetapi minggu validasi ini fokusnya adalah RAG, bukan belajar tooling baru.
- Nanti di fase MLOps (bulan 7–9) saat membahas reproducibility dan lockfile, saat itu tepat untuk mencoba PDM atau `uv`.

### Perintah-perintah kunci Hari 1

```
mkdir ai-engineer-journey && cd ai-engineer-journey
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install sentence-transformers
git add .
git commit -m "day 1: setup + dataset + first embeddings"
git push
```

**Penting:** `commit` saja masih tersimpan di laptopmu. `push` yang mengirimnya ke GitHub.

---

## 10. Dataset Hari Ini

- **Format:** JSONL — satu baris = satu dokumen. Mudah di-parse Python (`json.loads` per baris).
- **Isi:** 25 pasang Q&A tentang **konsep RAG/LLM** (dipilih karena sejalur niche; bonusnya kamu ikut menghafal konsep sambil membangun).
- **Plus:** 10 soal evaluasi (pertanyaannya diparafrase, bukan copy-paste) untuk menguji retrieval di Hari 5 — menguji retrieval sungguhan, bukan sekadar cocok string.

---

## 11. Glosarium RAG/LLM (dari Dataset)

| Istilah                      | Arti (1–2 kalimat)                                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **RAG**                      | Retrieval-Augmented Generation: menggabungkan pencarian dokumen dengan LLM, sehingga jawaban berdasar dokumen sendiri dan mengurangi halusinasi. |
| **Embedding**                | Representasi teks sebagai vektor angka yang menangkap makna. Teks yang maknanya mirip → vektornya berdekatan.                                    |
| **Vector database**          | Database yang menyimpan embedding dan mencari vektor tetangga terdekat dengan cepat. Contoh: Qdrant, pgvector, Chroma.                           |
| **Chunking**                 | Memotong dokumen panjang menjadi potongan kecil sebelum di-embedding, agar tiap potongan fokus satu topik dan retrieval lebih tepat.             |
| **Cosine similarity**        | Ukuran kemiripan dua vektor berdasarkan sudut di antaranya. Nilai 1 = sangat mirip. Dipakai untuk merangking hasil retrieval.                    |
| **LLM**                      | Large Language Model: model neural network besar yang dilatih memprediksi kata berikutnya. Contoh: Qwen, Llama, Mistral.                         |
| **Transformer**              | Arsitektur neural network berbasis attention, fondasi hampir semua LLM modern.                                                                   |
| **Token**                    | Unit teks terkecil yang diproses model (kata atau potongan kata). Model punya batas jumlah token (context window).                               |
| **Context window**           | Jumlah maksimum token yang bisa diterima model dalam satu prompt (instruksi + konteks + jawaban).                                                |
| **Halusinasi**               | Ketika LLM menjawab dengan percaya diri tetapi faktanya salah/tidak berdasar, karena model hanya memprediksi kata paling mungkin.                |
| **Prompt engineering**       | Teknik menyusun instruksi agar output LLM sesuai harapan: jelas, beri konteks cukup, sertakan contoh bila perlu.                                 |
| **Temperature**              | Parameter keacakan output. Rendah = deterministik & konsisten; tinggi = variatif & kreatif.                                                      |
| **Fine-tuning**              | Melatih ulang model pre-trained dengan data spesifik agar bagus di tugas tertentu.                                                               |
| **LoRA**                     | Low-Rank Adaptation: fine-tuning hemat yang hanya melatih sedikit parameter tambahan — murah dan cepat.                                          |
| **Reranking**                | Tahap kedua retrieval: kandidat dokumen diurutkan ulang dengan model yang lebih akurat.                                                          |
| **Quantization**             | Kompresi bobot model (mis. 16-bit → 4-bit) agar muat di RAM dan jalan di CPU dengan sedikit penurunan kualitas.                                  |
| **AI agent**                 | Sistem berbasis LLM yang bisa memakai tools, merencanakan langkah, dan bertindak mandiri untuk mencapai tujuan.                                  |
| **Evaluasi RAG**             | Mengukur dua hal: kualitas retrieval (dokumen relevan?) dan kualitas generation (jawaban benar & berdasar dokumen?).                             |
| **Guardrail**                | Aturan & pemeriksaan yang membatasi perilaku chatbot (menolak topik terlarang, memvalidasi output).                                              |
| **LangChain / LlamaIndex**   | Framework Python untuk membangun aplikasi LLM (RAG, agen) dengan komponen siap pakai.                                                            |
| **Ollama**                   | Tool untuk menjalankan LLM open-weight secara lokal, dengan API lokal di `http://localhost:11434`.                                               |
| **Top-p (nucleus sampling)** | Parameter sampling yang membatasi pilihan kata berikutnya pada subset kata berprobabilitas kumulatif tertinggi.                                  |
| **System prompt**            | Instruksi awal yang menetapkan peran, gaya, dan batasan perilaku chatbot sebelum percakapan dimulai.                                             |
| **Embedding model**          | Model khusus pembuat embedding. Contoh: sentence-transformers `all-MiniLM-L6-v2`.                                                                |
| **RAG vs fine-tuning**       | Saat data sering berubah, RAG lebih praktis: cukup update dokumen + index, tanpa melatih ulang model.                                            |

---

## 12. Cara Belajar yang Dipakai (Mastery-Based)

Progres perjalanan ini **berbasis penguasaan, bukan kalender**. Jadwal per bulan adalah patokan maksimum, bukan lantai. Setiap fase punya "gate" — kalau lulus, langsung lanjut, tidak perlu menunggu bulan berganti.

Setiap fase dinilai dari **3 bukti**, bukan dari lamanya belajar:

1. **Kode jalan** — proyek fase tersebut berfungsi end-to-end.
2. **Explain-back** — bisa menjelaskan konsep dengan kata sendiri tanpa buka catatan. Teknik "jelaskan supaya anak SD paham" (Feynman technique) adalah latihan persis untuk ini: kalau bisa menjelaskan ke anak SD, berarti benar-benar paham.
3. **Debug challenge** — bisa memperbaiki 1 error/bug yang diberikan.

**Gate Minggu 1–2 (validasi):** RAG v0 jalan (retrieval + jawaban Ollama), evaluasi ≥7/10, bisa menjelaskan embedding & RAG sendiri.

---

## 13. Besok: Hari 2 — Retrieval

Tugas besok: **retrieval** — mencari 3 dokumen paling mirip dengan pertanyaan. Langkahnya:

1. Load pasangan file kemarin (`embeddings.npy` + `docs.json`).
2. Embedding pertanyaan (wajib model yang sama!).
3. Hitung cosine similarity ke semua dokumen sekaligus (dot product, karena semua vektor sudah ternormalisasi).
4. Ambil 3 skor tertinggi.

Cara baca skor:

- ~0,8–1,0 → sangat relevan (jawaban hampir pasti ada di dokumen itu)
- ~0,4–0,7 → relevan, masih nyambung
- ~0,0–0,3 → tidak nyambung

---
