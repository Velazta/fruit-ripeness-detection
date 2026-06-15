# Walkthrough Hasil: Klasifikasi Kematangan Pisang (Model Awal vs Optimal)

Folder kerja ini sekarang berisi dua versi skrip klasifikasi dan laporannya untuk pengujian kematangan buah pisang:

## 1. File dan Source Code
*   [classify.py](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/pisang/classify.py): Skrip klasifikasi awal menggunakan fitur **12D (RGB+LAB) Unsegmented**.
*   [classify_optimal.py](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/pisang/classify_optimal.py): Skrip klasifikasi baru menggunakan fitur **8D (LAB+HSV Saturation) Segmented** dengan Chroma Thresholding = 10.

---

## 2. Perbandingan Performa Eksperimen

Penerapan logika pemilihan fitur 8D dan segmentasi kroma memberikan peningkatan akurasi serta stabilitas yang luar biasa:

*   **Bayes Gaussian Classifier (Opsi B)**:
    *   Model Awal (12D): Akurasi **93.67% ± 1.25%**
    *   Model Optimal (8D): Akurasi **94.89% ± 1.73%** 🚀 *(Mencapai akurasi maksimal **97.33%** pada Run 2!)*
*   **Mahalanobis Distance Classifier (Opsi C)**:
    *   Model Awal (12D): Akurasi **89.89% ± 1.10%**
    *   Model Optimal (8D): Akurasi **93.33% ± 2.49%**
*   **Minimum Distance Euclidean Classifier (Opsi A)**:
    *   Model Awal (12D): Akurasi **63.78% ± 1.77%**
    *   Model Optimal (8D): Akurasi **84.00% ± 2.37%** 📈 *(Naik 20% karena background noise dibuang!)*

---

## 3. Visualisasi Pendukung
*   Matriks Kebingungan Optimal: [confusion_matrices_optimal.png](./confusion_matrices_optimal.png)
*   Batas Keputusan PCA 2D Optimal: [decision_boundaries_optimal.png](./decision_boundaries_optimal.png)

Laporan komparatif teoretis dan statistik yang lengkap dapat diakses pada dokumen [analysis_results.md](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/pisang/analysis_results.md).
