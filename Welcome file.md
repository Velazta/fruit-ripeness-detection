
## 🍎 Algoritma Klasifikasi Kematangan Buah

### Alur Besar Sistem

```
📷 Input Gambar Buah
        ↓
🎨 Ekstraksi Fitur (warna RGB)
        ↓
📊 Masuk ke Classifier
        ↓
🏷️ Output: Unripe / Ripe / Rotten
```

---

### Tahap 1 — Ambil Fitur dari Gambar

Setiap piksel gambar punya nilai warna **R, G, B** (0–255):

```
Piksel gambar:
┌─────────┬─────────┬─────────┐
│   R     │   G     │   B     │
│ (Merah) │ (Hijau) │ (Biru)  │
└─────────┴─────────┴─────────┘

Contoh buah pisang:
  🟢 Belum matang → R rendah, G tinggi, B rendah
  🟡 Matang       → R tinggi, G tinggi, B rendah  
  🟤 Busuk        → R sedang, G rendah, B rendah
```

**Yang diekstrak** dari satu gambar:
```
rata_rata_R  = rata-rata semua piksel merah
rata_rata_G  = rata-rata semua piksel hijau
rata_rata_B  = rata-rata semua piksel biru
std_R        = penyebaran warna merah
std_G        = penyebaran warna hijau
std_B        = penyebaran warna biru
```

Sehingga **1 gambar → 1 vektor fitur**:
```
x = [rata_rata_R, rata_rata_G, rata_rata_B, std_R, std_G, std_B]
```

---

### Tahap 2 — Training (Hitung Statistik Tiap Kelas)

Dari dataset gambar yang sudah dilabeli:

```
Kelas UNRIPE (50 gambar):
  mean_unripe = [80, 160, 70, 15, 20, 12]
                   ↑   ↑   ↑
                   R   G   B  → hijau mendominan

Kelas RIPE (50 gambar):
  mean_ripe = [200, 190, 50, 25, 18, 15]
                  ↑   ↑   ↑
                  R   G   B  → kuning mendominan

Kelas ROTTEN (50 gambar):
  mean_rotten = [120, 80, 50, 30, 25, 20]
                   ↑   ↑   ↑
                   R   G   B  → gelap/cokelat
```

---

### Tahap 3 — Klasifikasi

#### **Opsi A: Minimum Distance Classifier** (paling sederhana)

```python
# Hitung jarak Euclidean ke tiap kelas
d_unripe  = || x - mean_unripe  ||
d_ripe    = || x - mean_ripe    ||
d_rotten  = || x - mean_rotten  ||

# Pilih yang paling dekat
if d_unripe paling kecil:
    hasil = "UNRIPE (Belum Matang)"
elif d_ripe paling kecil:
    hasil = "RIPE (Matang)"
else:
    hasil = "ROTTEN (Busuk)"
```

**Contoh hitung:**
```
Gambar buah yang diuji → fitur x = [190, 185, 55, 22, 16, 14]

d_unripe  = √((190-80)² + (185-160)² + (55-70)² + ...) = 115.2
d_ripe    = √((190-200)² + (185-190)² + (55-50)² + ...) =  12.7  ← paling kecil
d_rotten  = √((190-120)² + (185-80)² + (55-50)² + ...)  = 130.5

Hasil: RIPE ✅
```

---

#### **Opsi B: Bayes Decision Classifier** (lebih akurat)

```python
# Hitung probability tiap kelas
p_unripe = P(x | unripe) × P(unripe)
p_ripe   = P(x | ripe)   × P(ripe)
p_rotten = P(x | rotten) × P(rotten)

# Pilih probability tertinggi
hasil = argmax(p_unripe, p_ripe, p_rotten)
```

Menggunakan **distribusi Gaussian**:
```
P(x|kelas) = (1/√(2π|C|)) × exp(-½ (x-m)ᵀ C⁻¹ (x-m))

dimana:
  m = mean vektor tiap kelas
  C = kovarians matrix tiap kelas
```

---

#### **Opsi C: Mahalanobis Distance** (paling tepat)

```
d_mahalanobis(x) = (x - m)ᵀ × C⁻¹ × (x - m)

→ Mempertimbangkan penyebaran data tiap dimensi
→ Lebih akurat dari Euclidean jika data tidak seragam
```

---

### Tahap 4 — Decision Boundary (Visual)

```
        R (Merah)
        ↑
   Unripe│        ╱ Ripe
   (Hijau)│      ╱  (Kuning)
        │    ╱
        │  ╱
        │╱
        ┼──────────→ G (Hijau)
       ╱│
  Rotten│
  (Cokelat)
```

Garis di atas = **decision boundary** — pemisah antar kelas.

---

### Kode Python Lengkap (Minimum Distance)

```python
import cv2
import numpy as np

# ========== TAHAP 1: EKSTRAKSI FITUR ==========
def extract_features(image_path):
    """Ambil fitur warna dari gambar"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))  # resize seragam
    
    # Hitung rata-rata warna
    rata_rata = np.mean(img, axis=(0, 1))  # [R, G, B]
    
    # Hitung standar deviasi
    std_dev = np.std(img, axis=(0, 1))     # [std_R, std_G, std_B]
    
    return np.concatenate([rata_rata, std_dev])

# ========== TAHAP 2: TRAINING ==========
# Contoh data training (dari dataset)
# Format: [rata_R, rata_G, rata_B, std_R, std_G, std_B]
unripe_data = np.array([
    [80, 160, 70, 15, 20, 12],
    [75, 155, 65, 18, 22, 14],
    [85, 165, 72, 14, 19, 11],
    # ... tambah dari 50 gambar unripe
])

ripe_data = np.array([
    [200, 190, 50, 25, 18, 15],
    [195, 185, 55, 22, 20, 13],
    [205, 195, 48, 28, 16, 14],
    # ... tambah dari 50 gambar ripe
])

rotten_data = np.array([
    [120, 80, 50, 30, 25, 20],
    [115, 75, 45, 28, 23, 18],
    [125, 85, 55, 32, 27, 22],
    # ... tambah dari 50 gambar rotten
])

# Hitung mean tiap kelas
mean_unripe = np.mean(unripe_data, axis=0)
mean_ripe   = np.mean(ripe_data, axis=0)
mean_rotten = np.mean(rotten_data, axis=0)

print(f"Mean Unripe : {mean_unripe}")
print(f"Mean Ripe   : {mean_ripe}")
print(f"Mean Rotten : {mean_rotten}")

# ========== TAHAP 3: KLASIFIKASI ==========
def classify(image_path):
    """Klasifikasi kematangan buah"""
    x = extract_features(image_path)
    
    # Hitung Euclidean Distance
    d_unripe = np.linalg.norm(x - mean_unripe)
    d_ripe   = np.linalg.norm(x - mean_ripe)
    d_rotten = np.linalg.norm(x - mean_rotten)
    
    print(f"\nJarak ke Unripe : {d_unripe:.2f}")
    print(f"Jarak ke Ripe   : {d_ripe:.2f}")
    print(f"Jarak ke Rotten : {d_rotten:.2f}")
    
    # Pilih yang paling dekat
    distances = {
        'Unripe (Belum Matang)': d_unripe,
        'Ripe (Matang)': d_ripe,
        'Rotten (Busuk)': d_rotten
    }
    
    hasil = min(distances, key=distances.get)
    return hasil

# ========== TAHAP 4: UJI ==========
gambar_uji = "pisang_saya.jpg"
hasil = classify(gambar_uji)
print(f"\nHasil Klasifikasi: {hasil}")
```

---

### Ringkasan Alur

```
┌─────────────────────────────────────────────────┐
│              ALUR KLASIFIKASI                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. INPUT    → Foto buah dari kamera/HP         │
│       ↓                                          │
│  2. PREPROCESS → Resize, konversi RGB           │
│       ↓                                          │
│  3. FITUR    → Ekstrak rata-rata R,G,B + std   │
│       ↓                                          │
│  4. TRAINING → Hitung mean tiap kelas           │
│       ↓                                          │
│  5. CLASSIFY → Euclidean Distance / Bayes       │
│       ↓                                          │
│  6. OUTPUT   → Unripe / Ripe / Rotten           │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

### Pertanyaan di Slide Kuliah yang Cocok

| Konslide | Penerapan di Tugas |
|----------|--------------------|
| Slide 01–02 (Min Distance) | Hitung jarak fitur warna ke mean tiap kelas |
| Slide 03 (Mahalanobis) | Jika warna R,G,B punya variance beda-beda |
| Slide 04 (Bayes) | Tambahkan prior probability (misal: 60% matang, 20% unripe, 20% rotten) |
| Slide 05 (Binary Feature) | Threshold: R > 150 = "kuning", R ≤ 150 = "tidak kuning" |
| Slide 06 (Perceptron) | Learning otomatis batas keputusan dari data |
| Slide 09 (Fibonacci + LBP) | Jika pakai tekstur buah, bukan hanya warna |

Mau saya buatkan **kode lengkap + dataset contoh** yang bisa langsung dijalankan?
