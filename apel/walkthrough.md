# Walkthrough Hasil: Klasifikasi Kematangan Apel (Model Awal vs Optimal)

Folder kerja ini sekarang berisi dua versi skrip klasifikasi dan laporannya untuk pengujian kematangan buah apel:

## 1. File dan Source Code
*   [classify.py](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/apel/classify.py): Skrip klasifikasi awal menggunakan fitur **12D (RGB+LAB) Unsegmented**.
*   [classify_optimal.py](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/apel/classify_optimal.py): Skrip klasifikasi baru menggunakan fitur **8D (LAB+HSV Saturation) Segmented** dengan Chroma Thresholding = 10.

---

## 2. Perbandingan Performa Eksperimen

Penerapan logika pemilihan fitur 8D dan segmentasi kroma memberikan peningkatan akurasi serta stabilitas yang luar biasa:

*   **Bayes Gaussian Classifier (Opsi B)**:
    *   Model Awal (12D): Akurasi **78.22% ± 1.50%**
    *   Model Optimal (8D): Akurasi **88.44% ± 0.42%** 🚀 *(Naik 10.22% dan variansi mengecil drastis!)*
*   **Mahalanobis Distance Classifier (Opsi C)**:
    *   Model Awal (12D): Akurasi **74.56% ± 3.19%**
    *   Model Optimal (8D): Akurasi **84.89% ± 3.00%**
*   **Minimum Distance Euclidean Classifier (Opsi A)**:
    *   Model Awal (12D): Akurasi **55.33% ± 3.86%**
    *   Model Optimal (8D): Akurasi **61.44% ± 3.94%**

---

## 3. Visualisasi Pendukung
*   Matriks Kebingungan Optimal: [confusion_matrices_optimal.png](./confusion_matrices_optimal.png)
*   Batas Keputusan PCA 2D Optimal: [decision_boundaries_optimal.png](./decision_boundaries_optimal.png)

Laporan komparatif teoretis dan statistik yang lengkap dapat diakses pada dokumen [analysis_results.md](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/apel/analysis_results.md).
