# Walkthrough Hasil: Klasifikasi Kematangan Jeruk (Model Awal vs Optimal)

Folder kerja ini sekarang berisi dua versi skrip klasifikasi dan laporannya untuk pengujian kematangan buah jeruk:

## 1. File dan Source Code
*   [classify.py](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/jeruk/classify.py): Skrip klasifikasi awal menggunakan fitur **12D (RGB+LAB) Unsegmented**.
*   [classify_optimal.py](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/jeruk/classify_optimal.py): Skrip klasifikasi baru menggunakan fitur **8D (LAB+HSV Saturation) Segmented** dengan Chroma Thresholding = 10.

---

## 2. Perbandingan Performa Eksperimen

Penerapan logika pemilihan fitur 8D dan segmentasi kroma memberikan peningkatan akurasi serta stabilitas yang luar biasa:

*   **Bayes Gaussian Classifier (Opsi B)**:
    *   Model Awal (12D): Akurasi **90.22% ± 0.83%**
    *   Model Optimal (8D): Akurasi **93.22% ± 1.77%** 🚀 *(Mencapai akurasi maksimal **95.33%** pada Run 2!)*
*   **Mahalanobis Distance Classifier (Opsi C)**:
    *   Model Awal (12D): Akurasi **87.67% ± 1.70%**
    *   Model Optimal (8D): Akurasi **89.78% ± 2.53%**
*   **Minimum Distance Euclidean Classifier (Opsi A)**:
    *   Model Awal (12D): Akurasi **77.44% ± 0.68%**
    *   Model Optimal (8D): Akurasi **82.67% ± 0.47%**

---

## 3. Visualisasi Pendukung
*   Matriks Kebingungan Optimal: [confusion_matrices_optimal.png](./confusion_matrices_optimal.png)
*   Batas Keputusan PCA 2D Optimal: [decision_boundaries_optimal.png](./decision_boundaries_optimal.png)

Laporan komparatif teoretis dan statistik yang lengkap dapat diakses pada dokumen [analysis_results.md](file:///d:/KULIAH/CODINGAN/EXPERTSYSTEM/jeruk/analysis_results.md).
