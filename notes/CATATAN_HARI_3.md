# Catatan Belajar — Hari 3

**Tanggal:** Kamis, 15 Oktober 2026 (dilanjutkan dari sesi 14 Oktober)
**Tema:** Generation — menyambungkan Ollama, RAG v0 bisa menjawab
**Status:** ✅ Hari 3 selesai (pipeline end-to-end jalan + threshold bekerja + commit)

---

## Ringkasan Hari Ini (dalam 1 paragraf)

Hari ini RAG-mu berubah dari "mesin pencari" menjadi **chatbot yang menjawab**. Kamu belajar bahwa Ollama adalah server yang jalan di laptopmu sendiri, dan cara ngobrol dengan server itu adalah lewat **API** — mengirim HTTP request berisi JSON, lalu membaca jawabannya. Kamu merakit script `rag_chat.py` yang menggabungkan retrieval (Hari 2) + generation (Ollama), lengkap dengan threshold penolakan. Tiga pengujian menunjukkan pipeline bekerja — sekaligus menemukan dua failure mode nyata yang justru menjadi pelajaran diagnostik paling berharga sejauh ini.

---

## 1. Konsep: Ollama Itu Server di Laptopmu Sendiri

Waktu kamu install Ollama, dia diam-diam menjalankan **server** — program yang siaga menunggu permintaan. Alamatnya: `http://localhost:11434`.

- **localhost** = "komputer ini sendiri". Request-mu tidak pergi ke internet — berputar di dalam laptopmu saja. Makanya bisa offline dan gratis.
- **API** = "jendela layanan" server itu. Kamu tidak bisa asal ngomong ke Ollama; kamu harus lewat jendela resminya dengan format yang dia mengerti: **HTTP request berisi JSON**.
- **11434** = nomor port default pilihan developer Ollama (bisa diganti lewat environment variable `OLLAMA_HOST`, tapi bawaannya 11434).

### Analogi: drive-through

```
Kamu (Python)                          Ollama (dapur)
     |                                      |
     |--- slip pesanan (JSON) via POST ----->|   "requests.post(...)"
     |                                      |   dapur memasak jawaban...
     |<---- nampan jawaban (JSON) -----------|   "response.json()"
```

### Analogi: dapur – pelayan – meja makan

- **Ollama** = dapur (memasak jawaban, jalan di background)
- **Script Python-mu** = pelayan (mengambil jawaban dari dapur lewat API, lalu `print()` ke layar)
- **Terminal IDE** = meja makan (tempat jawaban disajikan — makanya kamu membacanya di terminal)

---

## 2. Anatomi Satu Panggilan API

Setiap panggilan API terdiri dari 4 komponen:

| Komponen | Isi | Artinya |
| --- | --- | --- |
| Metode | `POST` | "Saya mau **mengirim** data" (bukan mengambil) |
| Alamat | `http://localhost:11434/api/generate` | Jendela layanan Ollama untuk minta jawaban |
| Slip pesanan (JSON) | `{"model": ..., "prompt": ..., "stream": false}` | Model mana, pertanyaannya apa, mau jawaban langsung utuh |
| Timeout | `120` detik | "Kalau dapur tidak merespons dalam 2 menit, anggap gagal" |

### Isi slip pesanan

- **`"model": "qwen2.5:3b"`** → masak pakai resep yang mana.
- **`"prompt": "..."`** → pesanannya apa (pertanyaan + konteks + instruksi).
- **`"stream": false`** → "tunggu sampai matang semua, jangan diantar per suap". Kalau `true`, jawabannya datang kata-per-kata seperti efek mengetik ChatGPT — lebih rumit, jadi kita pakai `false` dulu.

### Tiga langkah di kode (pola universal semua API)

```python
import requests

# Langkah A — KIRIM: selipkan slip pesanan lewat jendela
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen2.5:3b", "prompt": "Halo, siapa kamu?", "stream": False},
    timeout=120,
)

# Langkah B — CEK: apakah dapur menerima pesanan? (200 = OK)
print(response.status_code)

# Langkah C — BACA: buka bungkusan JSON, ambil teks jawabannya
answer = response.json()["response"]
print(answer)
```

Alurnya selalu **A → B → C**: kirim, cek status, baca isi. Pola ini sama untuk semua API di dunia, bukan cuma Ollama.

**Catatan:** panggilan pertama bisa 10–30 detik (model sedang dimuat ke RAM). Panggilan berikutnya lebih cepat.

---

## 3. Dari Mana Tahu Endpoint Itu? (Pelajaran Verifikasi)

Endpoint `POST /api/generate` dan alamat `http://localhost:11434/api/generate` **tidak ditebak** — dibaca dari **dokumentasi resmi Ollama**: `github.com/ollama/ollama/blob/main/docs/api.md`. Di sana tertulis endpoint-nya dan contoh resminya memakai alamat yang sama persis.

### Pelajaran umumnya (penting untuk karirmu)

**Setiap tool punya "buku menu"-nya sendiri = dokumentasi API resmi.** Kalau nanti kamu pakai tool lain (Qdrant, LangChain, OpenAI API), langkah pertamamu selalu sama: buka docs resmi → cari halaman API → tiru contohnya. **Jangan hafal endpoint — hafal cara menemukannya.**

Dan insting "dari mana kamu tahu?" adalah **insting engineer yang benar**. Jangan telan mentah-mentah angka atau alamat yang dikasih orang (termasuk dari Zappy). Verifikasi ke sumber resmi.

---

## 4. Error-Error Hari Ini dan Artinya

### `ModuleNotFoundError: No module named 'requests'`

Library `requests` belum terinstall. Solusi: `pip install requests` (atau lihat bawah).

### Jebakan `pip` vs `python -m pip` (pelajaran penting!)

Setelah install, error tetap muncul. Penyebabnya: di komputermu ada **lebih dari satu Python** (Python sistem + Python di dalam `.venv`), dan perintah `pip` dan `python` menunjuk ke **Python yang berbeda**:

- `pip install requests` → menginstall ke Python A
- `python rag_chat.py` → dijalankan oleh Python B (yang tidak tahu-menahu soal install tadi)

**Solusi permanen — jadikan kebiasaan:**

```
python -m pip install requests
```

`python -m pip` artinya: "pakai **Python yang sama** dengan yang akan menjalankan script-ku." Dijamin install dan run selalu satu pasangan.

**Cara cek kalau kejadian lagi:**

```
which python
which pip
```

Kalau dua path-nya beda folder → tanda bahaya → langsung pakai `python -m pip`.

### File `__init__.py` (isi "dapus" library)

Setiap Python package punya file `__init__.py`: file yang **otomatis dibaca Python setiap kali kamu&#32;`import requests`**. Fungsinya seperti "daftar menu di pintu masuk" — dia mengimpor `post`, `get`, `ConnectionError`, dll dari file-file dalam, supaya kamu bisa langsung pakai `requests.post(...)`. (Baris `__cake__` di dalamnya hanya easter egg, tidak ada fungsi teknis.)

### `405 Method Not Allowed` (buka API di browser)

Membuka `http://localhost:11434/api/generate` di browser menghasilkan 405. **Ini bukan error yang perlu diperbaiki** — ini konfirmasi server hidup. Artinya: "alamatnya benar, tapi caranya salah."

- Browser address bar selalu mengirim **GET** = "lihat/ambil".
- `/api/generate` hanya menerima **POST** = "kirim data".

Ibarat datang ke drive-through lalu cuma melongo tanpa menyerahkan slip pesanan — ditolak. Cara POST manual dari terminal: `curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:3b","prompt":"Halo","stream":false}'`.

### Tabel error umum API

| Gejala | Penyebab | Solusi |
| --- | --- | --- |
| `ConnectionError` / tidak terhubung | Server Ollama mati | Jalankan `ollama serve` atau buka aplikasi Ollama |
| Status bukan 200 + kata "not found" | Model belum di-pull | `ollama pull qwen2.5:3b` |
| Timeout di panggilan pertama | Model sedang dimuat ke RAM | Tunggu, coba lagi |
| `405 Method Not Allowed` | Buka endpoint POST pakai browser (GET) | Bukan error — pakai Python/curl untuk POST |

---

## 5. Script `rag_chat.py` — RAG v0 yang Utuh

Alur script:

```
Pertanyaan → RETRIEVAL (top-3 dokumen) → THRESHOLD (skor < 0.5? tolak)
→ RAKIT PROMPT (konteks + instruksi + pertanyaan)
→ GENERATION (POST ke Ollama) → Jawaban ditampilkan + sumber dokumen
```

### Tiga hal baru di script ini

**1. Rakit prompt** — 3 dokumen hasil retrieval ditempel menjadi "Konteks", ditambah instruksi dan pertanyaan:

```
Jawablah pertanyaan berikut HANYA berdasarkan konteks di bawah ini.
Jika jawabannya tidak ada di konteks, katakan "Tidak ada di dokumen saya."
Gunakan Bahasa Indonesia.

Konteks:
...

Pertanyaan: ...
Jawaban:
```

Instruksi *"Jawablah HANYA berdasarkan konteks"* adalah **guardrail versi prompt**: cara paling sederhana mencegah LLM mengarang jawaban.

**2. Panggil Ollama lewat API lokal** — `POST http://localhost:11434/api/generate` dengan `model` + `prompt`. Pola yang sama dipakai semua aplikasi LLM production: HTTP request ke model server.

**3. Threshold diterapkan menjadi kode** — pelajaran kalibrasi Hari 2 langsung jadi implementasi:

```python
if scores[top3[0]] < THRESHOLD:  # THRESHOLD = 0.5
    print("Maaf, jawaban tidak ada di dokumen saya.")
```

Kalau skor dokumen terbaik di bawah 0.5, bot menolak menjawab daripada mengarang dari dokumen yang salah.

---

## 6. Tiga Pengujian Hari Ini (dan Dua Pelajaran Berharga)

### Tes 1 — "Berapa harga cabai hari ini?" ✅ SEMPURNA

Threshold menolak dengan benar (skor terbaik 0.426 < 0.5). Guardrail bekerja persis seperti dirancang.

### Tes 2 — "Apa itu LoRA?" ⚠️ retrieval miss

Dokumen yang ter-retrieve: #11 (prompt engineering), #23 (system prompt), #10 (halusinasi) — **dokumen LoRA tidak masuk top-3**. Model lalu menjawab "tidak ada di dokumen" — dan itu **jujur**, karena konteks yang diterimanya memang tidak berisi jawabannya.

**Pelajaran diagnostik (paling penting hari ini): bedakan retrieval failure vs generation failure.**

- **Retrieval failure** = dokumen yang benar tidak ketemu (kasus LoRA ini).
- **Generation failure** = dokumen benar ketemu, tapi model salah mengolah.

Obatnya beda: retrieval failure → perbaiki retrieval (reranking, chunking, model embedding lebih besar); generation failure → perbaiki prompt/model. **Salah diagnosis = memperbaiki bagian yang tidak rusak.**

### Tes 3 — "Jelaskan cara kerja RAG" ⚠️ halusinasi akronim

Model menulis **"Relevant-Aggregated Generation"** — SALAH. Yang benar "Retrieval-Augmented Generation", tertulis di dokumen #1... yang **tidak ter-retrieve** (yang ketemu malah #18, #15, #25). Model kecil (3B) + dokumen definisi tidak ada → dia mengarang kepanjangan akronim dengan percaya diri.

**Bukti hidup dua hal:** (1) RAG **mengurangi** halusinasi tapi tidak menghilangkannya; (2) kualitas jawaban = kualitas retrieval. Garbage in, garbage out.

---

---

## 8. Debug Challenge 🎯

Misteri: kenapa "Apa itu quantization?" (Hari 2) dapat dokumen benar di #1 (0.648), tapi "Apa itu LoRA?" gagal total? Cara menyelidiki: modifikasi `retrieve.py` supaya print **semua 25 skor terurut** untuk query LoRA, cari dokumen LoRA ada di peringkat berapa, lalu buat hipotesis. (Petunjuk: perhatikan betapa "umum" kata *quantization* vs betapa "langka" kata *LoRA* bagi model kecil.)

---

## 9. Alur Kerja & Progres

### Perintah penutup Hari 3

```
pip freeze > requirements.txt
git add . && git commit -m "day 3: RAG v0 with Ollama generation" && git push
```

`pip freeze > requirements.txt` mengunci daftar library supaya environment bisa direplika. Kebiasaan wajib: setiap tambah library baru, freeze ulang.

### Status perjalanan

- **Streak:** 3 hari (Hari 1 ✅, Hari 2 ✅, Hari 3 ✅).
- **Gate Minggu 1–2** (pengingat): RAG v0 jalan ✅ (retrieval + jawaban Ollama), evaluasi ≥7/10 (besok), bisa menjelaskan embedding & RAG sendiri.

---

## 10. Besok: Hari 4 — Evaluasi Formal

Tugas besok: uji **10 pertanyaan parafrase** (dataset evaluasi) dan hitung skornya — target gate **≥7/10**. Dua failure hari ini akan berubah dari "feeling" menjadi **angka**, dan dari angka itu kita tahu persis apa yang perlu diperbaiki.

---