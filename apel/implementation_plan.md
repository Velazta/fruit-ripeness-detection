# Rencana Implementasi: Sistem Klasifikasi Kematangan Buah (Khusus Apel)

Tujuan dari tugas ini adalah melakukan analisis mendalam terhadap **3 metode klasifikasi** (Minimum Distance / Euclidean, Bayes Decision Classifier, dan Mahalanobis Distance) yang dijelaskan dalam `Welcome file.md`, dengan menggunakan dataset buah yang dikhususkan pada **Apel** dengan 3 kategori:
- **Rotten** (Busuk): Menggunakan subfolder `rottenapples`
- **Unripe** (Belum Matang): Menggunakan subfolder `unripe apple`
- **Ripe** (Matang): Menggunakan subfolder `freshapples`

Proses training akan menggunakan **100 sampel acak per kategori** dari dataset apel yang tersedia, dan pengujian akan dilakukan menggunakan dataset pengujian (*test set*) untuk mengukur performa masing-masing metode secara komparatif.

---

## Metodologi Penelitian & Pemodelan

### 1. Ekstraksi Fitur Warna (6 Dimensi)
Untuk setiap gambar buah, kita akan mengekstrak vektor fitur 6-dimensi:
$$x = [\mu_R, \mu_G, \mu_B, \sigma_R, \sigma_G, \sigma_B]$$
- $\mu_R, \mu_G, \mu_B$: Rata-rata intensitas warna untuk channel Red, Green, dan Blue (setelah dikonversi dari BGR ke RGB).
- $\sigma_R, \sigma_G, \sigma_B$: Standar deviasi intensitas warna untuk masing-masing channel.

### 2. Formulasi Matematika dari Tiga Klasifikator

#### Opsi A: Minimum Distance Classifier (Euclidean Distance)
Klasifikator ini mengukur jarak garis lurus (Euclidean) dari vektor fitur uji $x$ ke rata-rata vektor fitur training dari tiap kelas $k$:
$$d_{\text{Euclidean}}(x, m_k) = \sqrt{(x - m_k)^T (x - m_k)} = \sqrt{\sum_{i=1}^6 (x_i - m_{k,i})^2}$$
Prediksi kelas adalah kelas dengan jarak terkecil:
$$\hat{y}_A = \arg\min_{k} d_{\text{Euclidean}}(x, m_k)$$

#### Opsi B: Bayes Decision Classifier
Mengasumsikan fitur data tiap kelas berdistribusi Gaussian Multivariat $\mathcal{N}(m_k, C_k)$. Kita memprediksi kelas dengan probabilitas posterior tertinggi. Dengan prior seragam $P(k) = \frac{1}{3}$, aturan keputusannya setara dengan memaksimalkan fungsi diskriminan log-likelihood:
$$g_k(x) = -\frac{1}{2} \ln |C_k| - \frac{1}{2} (x - m_k)^T C_k^{-1} (x - m_k)$$
Prediksi kelas:
$$\hat{y}_B = \arg\max_{k} g_k(x)$$
*Catatan:* Untuk menjamin stabilitas numerik jika matriks kovarians $C_k$ mendekati singular, kita menambahkan regularisasi Ridge (Shrinkage) pada diagonal matriks: $C_k \leftarrow C_k + 10^{-5} I$.

#### Opsi C: Mahalanobis Distance Classifier
Mengukur jarak dari vektor fitur $x$ ke rata-rata kelas $m_k$, dengan mempertimbangkan variansi dan korelasi antar fitur melalui matriks kovarians $C_k$:
$$d_{\text{Mahalanobis}}(x, m_k) = \sqrt{(x - m_k)^T C_k^{-1} (x - m_k)}$$
Prediksi kelas:
$$\hat{y}_C = \arg\min_{k} d_{\text{Mahalanobis}}(x, m_k)$$

---

## Rencana Perubahan

Kita akan membuat skrip Python di workspace untuk mengotomatisasi proses ini:
- `classify.py`: Skrip utama yang melakukan ekstraksi fitur, sampling acak dataset apel, training, klasifikasi 3 metode, evaluasi metrik, dan pembuatan visualisasi.

### [NEW] [classify.py](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/classify.py)
Skrip ini akan berisi:
1. Membaca gambar dari folder dataset apel dan mengekstrak fitur.
2. Memilih secara acak 100 gambar per kategori (`rottenapples`, `unripe apple`, `freshapples`) untuk training.
3. Melakukan training dengan menghitung $m_k$ dan $C_k$ untuk masing-masing kelas.
4. Mengevaluasi performa menggunakan dataset pengujian (`test`).
5. Menghitung akurasi, presisi, recall, F1-score, dan membuat matriks kebingungan (*confusion matrix*).
6. Melakukan visualisasi decision boundary dalam ruang fitur 2D (misalnya PCA 2D dari fitur 6D) untuk memperlihatkan bagaimana ketiga algoritma membagi ruang klasifikasi.

---

## Rencana Verifikasi

### Pengujian Otomatis
Kita akan menjalankan skrip `classify.py` secara langsung menggunakan Python. Skrip tersebut akan:
- Melaporkan statistik training untuk setiap kategori.
- Menguji performa ketiga opsi klasifikasi pada test set apel.
- Menghasilkan gambar matriks kebingungan dan decision boundary.
- Menyimpan hasil analisis ke file markdown laporan.

### Pengujian Manual
Kita akan memverifikasi hasil visualisasi decision boundary dan performa masing-masing model untuk memastikan tidak ada kesalahan pembagian matriks kovarians atau perhitungan jarak Mahalanobis.
