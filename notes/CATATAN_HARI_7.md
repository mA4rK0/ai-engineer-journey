# Catatan Belajar — Hari 7

**Tanggal:** Minggu, 25 Oktober 2026
**Tema:** Docker — membungkus API menjadi kontainer yang portable
**Status:** ✅ Hari 7 selesai dari sisi pembelajaran (Dockerfile + prosedur + diagnosis kegagalan build); verifikasi run/push menyusul setelah rebuild dengan Python 3.12

---

## Ringkasan Hari Ini (dalam 1 paragraf)

Hari ini RAG-mu dibungkus menjadi kontainer Docker: aplikasi + seluruh lingkungannya (versi Python, library, data) dikemas dalam satu image yang berjalan identik di mana pun. Kamu belajar anatomi Dockerfile baris per baris, perbedaan image vs container, perintah build/run/tag/push, perbedaan Docker (tool lokal) vs Docker Hub (gudang publik), jebakan `localhost` di dalam kontainer, dan — lewat satu kegagalan build yang sangat informatif — aturan emas bahwa versi Python image harus cocok dengan environment tempat `requirements.txt` dibuat.

---

## 1. Konsep: Kenapa Docker Ada?

### Penyakit "di laptopku jalan kok!"

Sampai Hari 6, API-mu jalan **di laptopmu** — dengan Python-mu, library-mu, folder-mu. Di laptop orang lain? Belum tentu. Versi Python beda, library beda, path beda — aplikasi bisa gagal dengan cara yang membingungkan.

### Obatnya: kontainer

Docker membungkus aplikasi **plus seluruh lingkungannya** ke dalam satu paket yang berjalan identik di mana pun. Janjinya: kalau jalan di kontainermu, jalan di mana pun.

### Tiga istilah yang jangan tertukar

| Istilah | Artinya | Analogi |
| --- | --- | --- |
| **Dockerfile** | Resep tertulis (file teks berisi instruksi) | Resep masakan |
| **Image** | Hasil build dari resep — paket siap jalan (read-only) | Cetakan kue |
| **Container** | Image yang sedang **dijalankan** (bisa banyak dari satu image) | Kue yang sedang dipanggang dari cetakan itu |

Satu image bisa dijalankan menjadi banyak container — sama seperti satu cetakan bisa memanggang banyak kue.

---

## 2. Bedah Dockerfile (Baris per Baris)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `FROM python:3.12-slim` — bahan dasar

Image dimulai dari **base image**: Linux ramping + Python 3.12 siap pakai. `slim` = tanpa paket tambahan yang tidak perlu (image lebih kecil). Versi **3.12** dipilih karena alasan di Bagian 7 (kegagalan build).

### `WORKDIR /app` — ruang kerja di dalam kontainer

Artinya: "di dalam rumah kontainer ini, jadikan folder `/app` sebagai ruang kerja utama."

- **Kontainer punya sistem file sendiri**, terpisah dari laptopmu. `/app` tidak ada di laptopmu — dia ada di dalam kontainer.
- Kalau `/app` belum ada, Docker **membuatnya otomatis** — tidak perlu `mkdir`.
- Setelah baris ini, semua perintah bekerja di dalam `/app`: `COPY . .` menyalin ke `/app`, `CMD` menjalankan dari `/app` (makanya `api:app` ditemukan).
- `/app` itu **nama konvensi**, bukan keharusan (bisa `/code`, `/srv`) — tapi komunitas sudah sepakat pakai `/app` supaya semua orang langsung paham.

Skemanya:

```
Laptop (Windows):  D:\programming-files\AI\ai-engineer-journey\
                              │  COPY . .
                              ▼
Kontainer (Linux): /app/api.py, /app/data/..., /app/requirements.txt
```

### `COPY requirements.txt .` + `RUN pip install ...` — install DULU sebelum copy kode

Urutan ini disengaja, dan alasannya adalah **layer caching**: Docker membangun image per-layer dan men-cache tiap layer. Kalau kode berubah tapi `requirements.txt` tidak, layer install dipakai ulang → build berikutnya jauh lebih cepat. Kalau urutannya dibalik (copy kode dulu), setiap perubahan kode memaksa install ulang — buang waktu.

Dua instalasi di baris ini:

1. **torch versi CPU dulu** — jauh lebih kecil dari versi GPU/CUDA (yang tidak kita pakai, karena kontainer jalan di CPU).
2. **Sisanya dari&#32;`requirements.txt`**.

### `COPY . .` — salin kode + data

Menyalin seluruh isi folder proyek (kecuali yang diabaikan `.dockerignore`) ke `/app`.

### `EXPOSE 8000` — dokumentasi port

Menyatakan port yang dipakai aplikasi. Ini dokumentasi — **bukan** membuka port ke luar (itu tugas `-p` di `docker run`).

### `CMD [...]` — perintah menjalankan server

`--host 0.0.0.0` = dengarkan koneksi dari **luar** kontainer. Tanpa ini, API hanya bisa diakses dari dalam kontainer sendiri — percuma.

---

## 3. `.dockerignore` vs `.gitignore` (Jangan Tertukar!)

Keduanya file "pengabaian", tapi untuk gudang yang berbeda:

| File | Mengabaikan untuk | Isi pentingnya |
| --- | --- | --- |
| **`.dockerignore`** | `docker build` (apa yang disalin ke image) | `.venv` (binary Windows — rusak & tidak berguna di Linux), `.git` |
| **`.gitignore`** | `git add` (apa yang di-commit ke GitHub) | `.venv/` (ribuan file library — besar dan tidak perlu, karena bisa install ulang dari `requirements.txt`), `__pycache__/`, `.env` |

Aturan praktis: **`.venv`&#32;tidak pernah dikirim ke mana-mana** — tidak ke image, tidak ke repo.

---

## 4. Perintah-Perintah Docker

| Perintah | Artinya | Keluar dari laptop? |
| --- | --- | --- |
| `docker --version` | Cek Docker terinstall | Tidak |
| `docker info` | Info daemon + (kalau login) baris `Username:` | Tidak |
| `docker build -t mini-rag .` | Masak image dari resep, beri nama `mini-rag`, simpan **lokal** | ❌ Tidak |
| `docker images` | Daftar image lokal | Tidak |
| `docker run -p 8000:8000 -e KUNCI=NILAI mini-rag` | Hidupkan container dari image lokal | ❌ Tidak |
| `docker tag mini-rag namamu/mini-rag` | Tempelkan label nama kedua (sesuai akun) pada image yang sama — tidak meng-copy apa-apa | ❌ Tidak |
| `docker login` | Login ke Docker Hub | ✅ (otentikasi) |
| `docker push namamu/mini-rag` | Kirim image ke Docker Hub | ✅ Ya |

### Dua flag di `docker run` yang wajib dipahami

- **`-p 8000:8000`** = "colokkan port 8000 laptopku ke port 8000 kontainer". Tanpa ini, API terkunci di dalam kontainer.
- **`-e OLLAMA_URL=...`** = set environment variable di dalam kontainer (lihat Bagian 5).

### Docker (tool lokal) vs Docker Hub (gudang publik)

`build`, `run`, `tag` bekerja dengan Docker lokal saja — gudang tidak terlibat. Hanya `push` (dan `login`) yang menyentuh internet. Analogi: `build` = masak dan simpan di kulkas sendiri; `run` = buka warung di teras sendiri; `push` = kirim bungkusan ke gudang pusat supaya orang lain bisa membuka cabang.

---

## 5. Jebakan `localhost` di Dalam Kontainer (Paling Penting!)

### Masalahnya

Ollama jalan di **laptop** (host). Di dalam kontainer, `localhost` artinya **kontainer itu sendiri** — bukan laptopmu. Kalau `api.py` tetap memakai `http://localhost:11434`, kontainer akan mencari Ollama di dalam dirinya sendiri dan gagal.

### Solusinya: konfigurasi lewat environment variable

Di `api.py`, tambahkan setelah import:

```python
import os
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
```

Dan ubah pemanggilan Ollama menjadi:

```python
r = requests.post(f"{OLLAMA_URL}/api/generate", ...)
```

Lalu saat menjalankan kontainer, set variabelnya ke alamat khusus host:

```
docker run -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 mini-rag
```

`host.docker.internal` = alamat khusus yang disediakan Docker untuk menunjuk ke mesin host dari dalam kontainer.

**Prinsipnya (production-grade):** satu kode, dua lingkungan. Di luar Docker, variabel tidak di-set → default `localhost` dipakai (perilaku lama). Di dalam Docker, variabel di-set → alamat host dipakai. Kode tidak diubah-ubah per lingkungan; yang berubah hanya konfigurasinya.

---

## 6. Debugging Docker: Daemon Dulu, Perintah Belakangan

### Error: `failed to connect to the docker API ... check if the daemon is running`

Artinya: perintahmu tidak bisa menemukan mesin Docker sama sekali. **Docker Desktop sedang tidak jalan.** Ini bukan salah perintahnya — tag `marko2506/mini-rag` formatnya sudah benar.

Urutan perbaikan:

1. Jalankan Docker Desktop dari Start menu, tunggu status "Running" (ikon paus berhenti beranimasi, 30–60 detik).
2. Tes: `docker info` — keluar info tanpa error = daemon hidup.
3. Cek image ada: `docker images` — kalau `mini-rag` tidak ada, berarti build belum pernah berhasil → `docker build -t mini-rag .` dulu.
4. Ulangi: `docker tag` lalu `docker push`.

**Pola debuggingnya:** error seperti ini hampir selalu berarti "aplikasinya belum dibuka", bukan "perintahnya salah". Cek daemon dulu, baru curigai perintah.

### Cara mengecek username Docker Hub

- Terminal (sudah login): `docker info` → cari baris `Username:`.
- Browser: hub.docker.com → pojok kanan atas setelah login.
- Belum punya akun: daftar gratis dulu di hub.docker.com — username saat registrasi itulah yang dipakai di `docker tag`.

---

## 7. Kegagalan Build: networkx 3.7 vs Python 3.11 (Pelajaran Termahal Hari Ini)

### Gejala

Build berjalan 200 detik, layer 1–3 berhasil (base image ter-download, torch CPU 196MB terpasang), lalu gagal di langkah install requirements:

```
ERROR: No matching distribution found for networkx==3.7
```

### Diagnosis

`requirements.txt` (hasil `pip freeze` di laptop Windows) meminta `networkx==3.7`. Tapi networkx 3.7 **hanya bisa dipasang di Python 3.12 ke atas** — sedangkan image memakai Python 3.11. Pip sudah berusaha ("Ignored the following versions that require a different python version") lalu menyerah.

### Perbaikan (satu baris)

```dockerfile
FROM python:3.12-slim
```

Lalu build ulang: `docker build -t mini-rag .`

### Aturan emas yang dipelajari dengan cara nyata

**Versi Python di image harus cocok dengan environment tempat&#32;`requirements.txt`&#32;dibuat.** Laptop Python 3.12 → image 3.12. Suatu hari requirements dibuat di Python 3.13 → image ikut 3.13. Ini penyakit "works on my machine" yang klasik — tapi terbalik: bukan kodenya yang beda, melainkan versi Python-nya.

### Praktik terbaik untuk jangka panjang

Jangan kirim hasil `pip freeze` mentah-mentah sebagai `requirements.txt` production: freeze ikut menyeret puluhan paket transitif (yang bisa konflik seperti ini). Tulis `requirements.txt` secara kurasi — hanya dependency langsung (`sentence-transformers`, `fastapi`, `uvicorn`, `requests`, `numpy`) — dan biarkan pip menyelesaikan versi transitifnya.

---

## 8. Alur Kerja Penutup Hari 7

### Ke GitHub (kode: `api.py`, `Dockerfile`, `.dockerignore`, `.gitignore`)

```
git status          # cek dulu; pastikan .venv TIDAK muncul
git add .
git commit -m "day 7: Docker"
git push
```

### Ke Docker Hub (image: `mini-rag`)

```
docker login
docker tag mini-rag marko2506/mini-rag
docker push marko2506/mini-rag
```

### Verifikasi

- GitHub: `Dockerfile` dan `api.py` versi terbaru ada di repo.
- Docker Hub: `hub.docker.com/r/marko2506/mini-rag` menampilkan image — dan dari laptop mana pun bisa ditarik dengan `docker pull marko2506/mini-rag`.

**Satu kalimat ringkasnya:** GitHub menyimpan resep dan kode, Docker Hub menyimpan kue yang sudah jadi.

---

## 9. Analogi Hari 7: Franchise 🍱

Hari 1–6 kamu membangun satu restoran yang bagus. Hari 7 kamu mengubahnya menjadi **paket franchise**: dapur, resep, peralatan, dan buku menu dikemas dalam satu boks kontainer yang bisa dikirim ke kota mana pun. Buka boksnya, colokkan listriknya (`docker run`), restoran langsung jualan — tanpa drama "di kota asal jalan, di kota baru gagal". Itulah janji Docker: **satu paket, jalan di mana saja.**

---

## 10. Berikutnya

- **Verifikasi Hari 7:** rebuild dengan Python 3.12 → `docker run` → `/health` OK dari browser → `/ask` menjawab (bukti kontainer menjangkau Ollama host) → `docker push marko2506/mini-rag`.
- **Backlog perbaikan:** query rewriting untuk kasus LoRA (query pendek + kata langka) — obat sudah teridentifikasi sejak Hari 4.
- **Ekspektasi jujur:** build pertama 5–15 menit (download Python + torch CPU) dan start pertama kontainer mengunduh model embedding (\~180MB) — keduanya normal dan hanya terjadi sekali.

---