<div align="center">

#  Neural Network with Scikit-learn
### Prediksi Diabetes Menggunakan Multi-Layer Perceptron (MLP)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-MLPClassifier-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

*Studi kasus penerapan Neural Network sederhana (MLP) untuk klasifikasi biner pada dataset medis, lengkap dengan hyperparameter tuning dan analisis feature importance.*

</div>

---

##  Ringkasan Proyek

Notebook ini merupakan bagian **opsional** dari materi Neural Network, yang menunjukkan bagaimana arsitektur *Multi-Layer Perceptron* (MLP) — dasar dari deep learning — dapat diimplementasikan dengan cepat menggunakan `MLPClassifier` dari **Scikit-learn**, tanpa perlu framework deep learning seperti TensorFlow/PyTorch.

Studi kasus yang digunakan adalah **prediksi diabetes** berdasarkan data medis pasien (Pima Indians Diabetes Dataset), dengan target memprediksi apakah seorang pasien positif diabetes (`Outcome`: 0 atau 1).

>  **Insight utama notebook ini:** untuk data tabular berukuran kecil–menengah, Neural Network *tidak selalu* menjadi pilihan terbaik dibanding algoritma ML konvensional (Random Forest, SVM, XGBoost) — baik dari segi akurasi maupun waktu training.

---

##  Dataset

| Detail | Keterangan |
|---|---|
| **Sumber** | [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) |
| **Target** | `Outcome` (0 = negatif diabetes, 1 = positif diabetes) |
| **Fitur** | `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age` |
| **Tipe Masalah** | Klasifikasi biner |

---

##  Alur Kerja (Workflow)

```
1. Import Data           → Load dataset diabetes.csv & cek missing value
2. Dataset Splitting     → Train-test split (80:20) dengan stratifikasi
3. Preprocessing         → ColumnTransformer + numeric pipeline (jcopml)
4. Modeling              → MLPClassifier + GridSearchCV
5. Feature Importance    → Mean score decrease analysis
6. Refinement            → Preprocessing lanjutan (polynomial + box-cox) 
                            pada fitur paling berpengaruh
```

###  Detail Tahapan

**1. Import & Eksplorasi Data**
Dataset dimuat dan diperiksa kelengkapannya menggunakan `plot_missing_value`, serta distribusi kelas target dicek untuk mendeteksi potensi *class imbalance*.

**2. Dataset Splitting**
Data dibagi menjadi data latih dan data uji (80:20) menggunakan `train_test_split` dengan `stratify=y` agar proporsi kelas tetap seimbang di kedua subset.

**3. Preprocessing Pipeline**
Menggunakan `ColumnTransformer` dan `num_pipe()` dari library `jcopml` untuk menangani fitur numerik secara otomatis (scaling & imputation).

**4. Modeling dengan MLPClassifier**
Model neural network dibangun dengan `MLPClassifier`, kemudian di-*tuning* menggunakan `GridSearchCV` (3-fold CV) pada parameter berikut:

| Parameter | Nilai yang Diuji | Fungsi |
|---|---|---|
| `alpha` | 0.0001, 0.0003, 0.001, 0.003 | Regularisasi (mencegah overfitting) |
| `hidden_layer_sizes` | (16,), (32,16), (32,16,8) | Arsitektur jaringan (1–3 hidden layer) |
| `learning_rate_init` | 0.001, 0.005, 0.01 | Kecepatan belajar model |
| `activation` | relu, logistic, tanh | Fungsi aktivasi neuron |

**5. Feature Importance**
Menggunakan `mean_score_decrease` untuk mengidentifikasi fitur paling berpengaruh terhadap prediksi model.

**6. Refinement**
Berdasarkan hasil feature importance, dilakukan preprocessing lanjutan (polynomial degree 2 + transformasi box-cox) khusus pada fitur `Glucose`, `BMI`, dan `SkinThickness`, lalu model di-tuning ulang untuk melihat peningkatan performa.

---

##  Tech Stack

- **Python 3** — bahasa pemrograman utama
- **Pandas & NumPy** — manipulasi & analisis data
- **Scikit-learn** — `MLPClassifier`, `GridSearchCV`, `Pipeline`, `ColumnTransformer`
- **jcopml** — utility pipeline (preprocessing, tuning params, feature importance, plotting)
- **luwiji** — visualisasi ilustrasi konsep neural network

---

##  Cara Menjalankan

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn jcopml luwiji

# 2. Pastikan dataset tersedia di:
#    data/diabetes.csv

# 3. Jalankan notebook
jupyter notebook "Part_2_-__Opsional__Neural_Network_with_Scikit-learn.ipynb"
```

---

## Catatan & Pembelajaran

- GridSearchCV pada MLP membutuhkan waktu komputasi lebih lama dibanding algoritma ML konvensional (RF, SVM, XGBoost), karena proses training neural network yang bersifat iteratif.
- Untuk data **tabular** berskala kecil, algoritma ML klasik sering kali menjadi pilihan yang lebih efisien dan kompetitif dibanding neural network.
- Neural network lebih unggul pada data berdimensi tinggi/tidak terstruktur (gambar, teks, audio) dibanding data tabular sederhana seperti pada studi kasus ini.

---

<div align="center">

*Bagian dari eksplorasi materi Neural Network — implementasi cepat menggunakan Scikit-learn.*

</div>
