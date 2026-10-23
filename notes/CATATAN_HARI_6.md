# Catatan Belajar — Hari 6

**Tanggal:** Jumat, 23 Oktober 2026 (dimulai 19 Oktober, dikerjakan kembali setelah minggu UTS)
**Tema:** FastAPI — RAG berubah dari script menjadi layanan API
**Status:** ✅ Hari 6 selesai (API jalan: `/health` sehat, `/ask` melayani end-to-end + commit)

---

## Ringkasan Hari Ini (dalam 1 paragraf)

Hari ini RAG-mu naik kelas: dari **script terminal** (`rag_chat.py`) menjadi **layanan API** (`api.py`) yang bisa ditanya oleh program lain lewat HTTP. Kamu belajar pola client-server dari sisi sebaliknya — dulu script-mu jadi *klien* yang bertanya ke Ollama, sekarang API-mu jadi *server* yang ditanya dunia luar. API-nya terverifikasi jalan (`/health` OK, `/ask` mengembalikan jawaban + sources + scores), dan satu pengujian LoRA memberi pelajaran diagnosis terakhir: membedakan penolakan threshold, penolakan jujur LLM, dan retrieval failure.

---

## 1. Konsep: Dari Script Menjadi Service

### Perbedaannya

|  | Script (`rag_chat.py`) | Service/API (`api.py`) |
| --- | --- | --- |
| Cara pakai | Dijalankan manusia di terminal | Ditanya program lain lewat HTTP |
| Model di-load | Tiap script dijalankan | **Sekali** saat server start, melayani ribuan request |
| Input/output | Ketik & baca di terminal | JSON masuk, JSON keluar |
| Siapa yang bisa pakai | Hanya kamu | Web, aplikasi mobile, bot, program apa pun |

### Kebalikan peran (insight Hari 6)

Di Hari 3, **kamu jadi klien**: script-mu mengirim `POST` ke server Ollama di `localhost:11434`. Hari ini **kamu jadi server**: program lain mengirim `POST` ke API-mu di `localhost:8000`. Polanya sama persis — yang bertukar hanya posisinya. Memahami dua sisi ini = memahami cara kerja hampir semua sistem AI production.

---

## 2. Bedah `api.py`

### Dua endpoint (jendela layanan)

**`GET /health`** — cek kesehatan server.

```json
{ "status": "ok", "documents": 25 }
```

Pola wajib semua API production: monitoring/load balancer bertanya "hidup nggak?" tanpa memicu kerja berat.

**`POST /ask`** — pipeline RAG lengkap dalam satu request.

- Request: `{ "question": "..." }`
- Response: `{ "answer": "...", "sources": [21, 10, 6], "scores": [3.096, 2.723, 2.617] }`

### Pydantic: kontrak resmi API

`AskRequest` dan `AskResponse` adalah **skema data** yang dideklarasikan sekali. FastAPI + Pydantic memvalidasi otomatis: request salah format ditolak dengan pesan error jelas — tanpa kode validasi manual. Ini "kontrak" yang membuat klien dan server tidak salah paham soal bentuk data.

### Load sekali, layani ribuan

Model embedding, cross-encoder, dan data di-load **satu kali saat server start**, bukan tiap pertanyaan. Ini beda fundamental script vs service — dan alasan API bisa cepat: pekerjaan mahal dilakukan di depan.

### Threshold di tempat yang benar (keputusan desain penting)

Di `/ask`, urutannya: tahap 1 (bi-encoder, 10 kandidat) → **threshold 0.5 dicek pada skor cosine tahap 1** → tahap 2 (rerank) → generation.

Kenapa threshold tidak memakai skor rerank? Karena skor cross-encoder adalah **raw logits** (skalanya beda, bisa negatif) — angka 0.5 hanya bermakna dalam skala cosine yang sudah dikalibrasi di Hari 2. Mencampur dua skala = bug yang diam-diam merusak. Detail kecil seperti ini yang membedakan engineer yang paham dari yang sekadar jalan.

### Dokumentasi otomatis

FastAPI membuatkan halaman **`/docs`** (Swagger UI) secara gratis: daftar endpoint, skema, dan formulir coba-interaktif langsung dari browser. Kamu tidak menulis satu baris pun untuk itu — itu hasil deklarasi skema yang rapi.

### Cara menjalankan

```
python -m pip install fastapi uvicorn
uvicorn api:app --reload
```

`uvicorn` adalah server yang menjalankan aplikasi FastAPI-mu (menjaga "loket" tetap terbuka dan menerima request).

---

## 3. Hasil Pengujian

### `/health` ✅

```json
{ "status": "ok", "documents": 25 }
```

Server hidup, 25 dokumen ter-load.

### `/ask` ✅ (pipeline) — dengan satu temuan

```json
{
  "answer": "Tidak ada di dokumen saya.",
  "sources": [21, 10, 6],
  "scores": [3.096, 2.723, 2.617]
}
```

Seluruh pipeline terbukti jalan: retrieval two-stage → threshold lolos → prompt terakit → Ollama menjawab → JSON terstruktur kembali. Temuannya dibedah di Bagian 4.

---

## 4. Bedah Kasus: Dua Lapis Penolakan dan Kasus LoRA

### Dua jenis "tidak ada" (jangan tertukar!)

|  | Penolakan threshold (API) | Penolakan jujur (LLM) |
| --- | --- | --- |
| Siapa yang menolak | Kode API-mu, sebelum LLM dipanggil | LLM sendiri, setelah membaca konteks |
| Pesan | "Maaf, jawaban tidak ada di dokumen saya." | "Tidak ada di dokumen saya." |
| Sources | **Kosong** `[]` | **Terisi** (dokumen yang dibaca LLM) |
| Arti | Skor cosine terbaik < 0.5 | Konteks lolos, tapi jawaban dinilai tidak ada di dalamnya |

Hasil `/ask` di atas adalah jenis kedua: threshold **lolos** (sources terisi), LLM membaca konteksnya dan memutuskan menolak. Sistem pengaman bekerja di **dua lapis**.

### Apakah penolakannya benar? — Jawaban ADA

Pertanyaan yang dikirim adalah soal **LoRA**. Jawabannya **ada** di dokumenmu sendiri: **dokumen #14 "Apa itu LoRA?"** — terbukti terukur di evaluasi Hari 5 (parafrase LoRA mendapat #14 di peringkat #1, skor 7.763).

Jadi diagnosisnya: **retrieval failure — ketiga kalinya, pola yang sama persis.**

- Retrieval tidak membawa dokumen #14 ke top-3 (yang datang: #21 Ollama, #10 halusinasi, #6 LLM — dokumen umum soal LLM).
- LLM menerima konteks yang salah, tidak menemukan LoRA di sana, lalu **dengan jujur** menolak.
- LLM-nya tidak salah; dia setia pada konteks. Yang salah adalah pengantarnya.

### Pola yang sekarang konsisten (sudah 3x teramati)

| Bentuk query | Hasil |
| --- | --- |
| Pendek + kata langka ("Apa itu LoRA?") | ❌ Gagal |
| Deskriptif ("Teknik fine-tuning apa yang hemat parameter?") | ✅ Berhasil #1 |

**Obatnya (sudah terpetakan sejak Hari 4): query rewriting** — sebelum retrieval, query pendek user ditulis ulang menjadi bentuk deskriptif oleh LLM (Ollama), meniru parafrase yang terbukti berhasil.

### Kesimpulan /ask (3 lapis)

1. API terbukti jalan penuh — Hari 6 sukses.
2. Sistem pengaman bekerja dua lapis — API tidak rusak, dia jujur.
3. Kebenaran penolakan terakhir bergantung pada retrieval — dan di kasus LoRA, retrieval-lah yang gagal, bukan generation.

---

## 5. Analogi Hari 6: Loket Drive-Through 🚗

Kemarin (`rag_chat.py`), pelanggan harus **masuk dan duduk di dalam dapur** (terminal) dan ngobrol langsung dengan pelayan. Hari ini restoranmu **membuka loket drive-through**:

| Elemen API | Analogi restoran |
| --- | --- |
| `api.py` | **Loket drive-through** — pintu resmi untuk pelanggan dari luar |
| `uvicorn` | Petugas loket yang siaga 24 jam menerima pesanan |
| `AskRequest` (Pydantic) | **Formulir pesanan baku** — kolom wajib: pertanyaan. Salah isi → formulir dikembalikan, dapur tidak terganggu |
| `/health` | Papan **"BUKA"** — "buka nggak? menunya ada berapa?" → "ok, 25" |
| `/ask` | Satu pesanan lengkap: formulir masuk → pelayan cari hidangan → koki masak → **nampan keluar** berisi jawaban + daftar bahan (sources) + tingkat keyakinan koki (scores) |
| `/docs` (Swagger) | **Buku menu interaktif** di loket — bisa coba pesan langsung dari sana |
| Model di-load sekali | **Mise en place** — dapur menyiapkan semua bahan di pagi hari, bukan motong bawang tiap pesanan |

Dan kasus LoRA dalam bahasa restoran: loketnya bekerja **sempurna** — formulir diterima, diproses, nampan keluar lengkap. Yang salah tetap pelayan pencicipnya: membawa hidangan referensi yang keliru, dan koki yang jujur berkata "bahan ini nggak ada". Gunanya loket: kalau ada yang salah, kamu tahu persis salahnya di **dapur**, di **pelayan**, atau di **pintu masuk**.

**Satu kalimat Hari 6:** kemarin pelanggan masuk ke dapurmu; sekarang dapurmu yang melayani dunia luar — lewat loket, dengan formulir, tanpa pelanggan menyentuh kompor.

---

## 6. Analogi Lengkap Hari 1–6: Restoran RAG 🍽️

Satu cerita utuh dari nol sampai punya loket:

### `embed_first.py` — Juru bumbu (Hari 1)

Sebelum restoran buka, juru bumbu menyiapkan **kartu rasa** untuk 25 hidangan: tiap hidangan dicicipi dan diberi 384 nilai rasa. Dikerjakan sekali di pagi hari. *(Teknis: 25 dokumen → vektor 384 dimensi →&#32;`embeddings.npy`&#32;+&#32;`docs.json`.)*

### `retrieve.py` — Pelayan pencicip (Hari 2)

Pelanggan bilang "mau yang manis seperti es krim". Pelayan mengubah keinginan itu jadi "rasa", membandingkannya dengan 25 kartu rasa, dan membawa **3 hidangan terdekat** + skor kemiripannya. *(Teknis: embedding query → cosine similarity → top-3.)*

### `test_ollama.py` — Tes kompor (Hari 3)

Sebelum masak beneran, koki menyalakan kompor 5 detik: "apinya nyala nggak?" Bukan masak — cuma memastikan dapur hidup. *(Teknis: 1 panggilan API sederhana untuk verifikasi koneksi.)*

### `rag_chat.py` — Restoran beroperasi (Hari 3)

Pelanggan pesan → pelayan cari 3 hidangan terdekat → **kalau tidak ada yang mirip, tolak dengan sopan** ("maaf, tidak ada di menu kami" = threshold 0.5) → kalau ada, serahkan ke **koki** (Ollama) untuk diracik jadi jawaban → hidangkan. Buka terus sampai pelanggan bilang `exit`. *(Teknis: retrieval + threshold + prompt assembly + generation.)*

### `evaluate.py` — Food critic misterius (Hari 4)

Kritikus datang menyamar dengan **10 pesanan yang dibahasakan berbeda** (parafrase). Dia menilai berapa dari 10 pesanan yang hidangan bawaannya benar. **Nilai: 8/10.** *(Teknis: top-3 accuracy atas 10 soal parafrase.)*

### `rerank.py` — Koki kepala pencicip ulang (Hari 5)

Restoran mempekerjakan koki kepala: pelayan (cepat tapi asal) membawa **10 kandidat**, koki kepala **mencicipi satu per satu dengan teliti** dan memilih 3 terbaik. Lambat, tapi akurat. *(Teknis: cross-encoder membaca pasangan query+dokumen bersamaan.)*

### `evaluate_rerank.py` — Kritikus datang lagi (Hari 5)

Kritikus yang sama kembali setelah ada koki kepala. **Nilai naik: 9/10.** Satu pesanan masih gagal — hidangannya tidak pernah sampai ke meja koki kepala karena pelayan membuangnya di pintu (recall bottleneck). *(Teknis: evaluasi two-stage retrieval.)*

### `api.py` — Loket drive-through (Hari 6)

Restoran membuka **loket resmi**: pelanggan (program lain) tidak perlu masuk dapur lagi — pesan lewat loket dengan formulir baku, nampan keluar berisi pesanan + daftar bahan + tingkat keyakinan koki. *(Teknis: FastAPI + Pydantic + uvicorn.)*

### File datanya

- `embeddings.npy` = **tumpukan kartu rasa**
- `docs.json` = **buku menu** (kartu ke-7 selalu milik hidangan ke-7)

### Satu kalimat untuk semuanya

**Hari 1 menyiapkan dapur, Hari 2 melatih pelayan, Hari 3 membuka restoran, Hari 4 mengundang kritikus, Hari 5 mempekerjakan koki kepala, Hari 6 membuka loket drive-through.**

---

## 7. Alur Kerja Penutup

```
pip freeze > requirements.txt
git add . && git commit -m "day 6: FastAPI" && git push
```

---

## 8. Berikutnya

- **Hari 7 — Docker:** API-mu dibungkus jadi kontainer — satu perintah, jalan di laptop mana pun tanpa drama install. Ini yang mengubah "jalan di laptopku" menjadi "jalan di mana saja", syarat proyek portofolio yang ter-deploy.
- **Backlog perbaikan:** query rewriting untuk kasus LoRA (query pendek + kata langka) — obat sudah teridentifikasi, tinggal dieksekusi dan diukur.

---