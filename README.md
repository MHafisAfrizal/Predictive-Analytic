# Laporan Proyek Machine Learning - Hafis Afrizal

## Domain Proyek
Readmisi pasien diabetes merupakan tantangan besar dalam sistem kesehatan, khususnya di Amerika Serikat. Sekitar 20-30% pasien diabetes kembali dirawat dalam 30 hari setelah keluar rumah sakit, menyebabkan biaya tahunan mencapai miliaran dolar dan menurunkan kualitas hidup pasien [1]. Masalah ini diperparah oleh kurangnya alat prediktif yang akurat untuk mengidentifikasi pasien berisiko tinggi, sehingga menghambat intervensi dini. Proyek ini bertujuan memprediksi risiko readmisi pasien diabetes menggunakan pendekatan regresi machine learning berdasarkan data klinis dari *Diabetes 130-US Hospitals for Years 1999-2008* ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)). Dengan prediksi yang akurat, rumah sakit dapat mengalokasikan sumber daya lebih efisien, mengurangi biaya, dan meningkatkan perawatan pasien.

**Referensi**:
[1] A. S. Ahmad, "Hospital readmissions among patients with diabetes," *Journal of Healthcare*, vol. 45, no. 3, pp. 123-130, 2020.

## Business Understanding
### Problem Statements
- Tingginya tingkat readmisi pasien diabetes dalam waktu <30 hari meningkatkan biaya operasional rumah sakit dan membebani sistem kesehatan.
- Kurangnya alat prediktif berbasis data menghambat rumah sakit dalam mengidentifikasi pasien berisiko tinggi untuk intervensi dini.

### Goals
- Mengembangkan model machine learning yang akurat untuk memprediksi risiko readmisi pasien diabetes, diukur dengan metrik MAE, MSE, dan R².
- Mengidentifikasi faktor klinis utama yang memengaruhi risiko readmisi untuk mendukung pengambilan keputusan klinis.

### Solution Statements
- Membandingkan performa tiga algoritma regresi (Regresi Linear, Random Forest, XGBoost) menggunakan metrik evaluasi MAE, MSE, dan R² untuk memilih model terbaik.
- Melakukan penyetelan hiperparameter pada Random Forest dan XGBoost menggunakan GridSearchCV untuk meningkatkan akurasi prediksi.

## Data Understanding
Dataset yang digunakan adalah subset 5000 sampel dari *Diabetes 130-US Hospitals for Years 1999-2008*, tersedia di UCI Machine Learning Repository: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). Dataset ini berisi catatan klinis pasien diabetes dengan 50 kolom, termasuk variabel target `readmitted` yang diubah menjadi skor risiko berkelanjutan (`risiko_readmisi`: 0 untuk 'NO', 0.5 untuk '>30', 1 untuk '<30').

### Variabel-Variabel
- **encounter_id**: Identifikasi unik kunjungan pasien (numerik).
- **patient_nbr**: Identifikasi unik pasien (numerik).
- **race**: Ras pasien (kategorikal: Caucasian, AfricanAmerican, Hispanic, Asian, Other, Unknown; 113 missing).
- **gender**: Jenis kelamin pasien (kategorikal: Male, Female, Unknown/Invalid).
- **age**: Kelompok usia pasien (kategorikal: [0-10), [10-20), ..., [90-100)).
- **weight**: Berat badan pasien (kategorikal: rentang berat atau Unknown; 4830 missing).
- **admission_type_id**: Jenis penerimaan (numerik: Emergency, Urgent, Elective, dll.).
- **discharge_disposition_id**: Status keluar pasien (numerik: Discharged to home, Transferred, dll.).
- **admission_source_id**: Sumber penerimaan (numerik: Physician Referral, Emergency Room, dll.).
- **time_in_hospital**: Lama tinggal di rumah sakit (numerik: hari).
- **payer_code**: Kode pembayar (kategorikal: MC, HM, SP, dll.; 1967 missing).
- **medical_specialty**: Spesialisasi dokter (kategorikal: InternalMedicine, Cardiology, dll.; 2435 missing).
- **num_lab_procedures**: Jumlah prosedur laboratorium (numerik).
- **num_procedures**: Jumlah prosedur non-laboratorium (numerik).
- **num_medications**: Jumlah obat yang diberikan (numerik).
- **number_outpatient**: Jumlah kunjungan rawat jalan sebelumnya (numerik).
- **number_emergency**: Jumlah kunjungan darurat sebelumnya (numerik).
- **number_inpatient**: Jumlah kunjungan rawat inap sebelumnya (numerik).
- **diag_1, diag_2, diag_3**: Kode diagnosis (kategorikal: kode ICD-9; 1, 16, 73 missing).
- **number_diagnoses**: Jumlah diagnosis yang dicatat (numerik).
- **max_glu_serum**: Hasil tes glukosa serum (kategorikal: None, Norm, >200, >300; 4740 missing).
- **A1Cresult**: Hasil tes HbA1c (kategorikal: None, Norm, >7, >8; 4154 missing).
- **metformin, repaglinide, ..., insulin, ...**: Status penggunaan obat diabetes (kategorikal: No, Steady, Up, Down).
- **change**: Perubahan pengobatan diabetes (kategorikal: Ch, No).
- **diabetesMed**: Apakah pasien menerima obat diabetes (kategorikal: Yes, No).
- **readmitted**: Status readmisi (kategorikal: NO, <30, >30; diubah ke `risiko_readmisi`).

### Exploratory Data Analysis (EDA)
- **Missing Values**: Kolom `weight` (96% missing), `medical_specialty` (48.7% missing), dan `payer_code` (39.3% missing) memiliki missing values signifikan, memerlukan penanganan khusus.
- **Duplikat**: Tidak ada data duplikat, menunjukkan kualitas data yang baik.
- **Distribusi**: Fitur numerik seperti `time_in_hospital` dan `num_medications` menunjukkan distribusi miring, memerlukan penanganan outlier.

## Data Preparation
Tahapan persiapan data dilakukan secara berurutan sesuai notebook untuk memastikan data bersih dan relevan:
1. **Penanganan Missing Values**:
   - Kolom kategorikal (`race`, `weight`, `payer_code`, `medical_specialty`) diisi dengan 'Unknown' untuk mempertahankan informasi.
   - Kolom `weight`, `payer_code`, dan `medical_specialty` dihapus karena missing values >39%, yang dapat mengurangi akurasi model.
2. **Penghapusan Duplikat**:
   - Tidak ada duplikat ditemukan, memastikan data unik.
3. **Penanganan Outlier**:
   - Winsorization (batas 5% ekor distribusi) diterapkan pada `time_in_hospital` dan `num_medications` untuk mengurangi dampak nilai ekstrem tanpa menghapus data.
4. **Rekayasa Fitur**:
   - `total_prosedur`: Jumlah dari `num_lab_procedures`, `num_procedures`, `number_outpatient`, `number_emergency`, dan `number_inpatient` untuk menangkap intensitas perawatan.
   - `kelompok_usia`: Usia dikelompokkan menjadi 'Muda' ([0-30)), 'Setengah Baya' ([30-60)), dan 'Senior' ([60-100)) untuk menyederhanakan analisis.
   - `risiko_readmisi`: Kolom `readmitted` diubah menjadi skor berkelanjutan (0, 0.5, 1) untuk pendekatan regresi.
5. **Pengkodean dan Skalasi**:
   - Variabel kategorikal (misalnya, `race`, `gender`, `kelompok_usia`) dienkode menggunakan `LabelEncoder` untuk mengubahnya menjadi numerik.
   - Fitur numerik diskalakan dengan `StandardScaler` untuk menormalkan distribusi dan meningkatkan performa model.
6. **Pemisahan Data**:
   - Data dibagi menjadi 80% pelatihan dan 20% pengujian dengan `random_state=42` untuk reproduktibilitas.

**Alasan Tahapan**:
- Penghapusan kolom dengan missing values tinggi mengurangi noise.
- Winsorization menjaga data outlier tetap relevan.
- Rekayasa fitur meningkatkan relevansi prediktif.
- Pengkodean dan skalasi memastikan kompatibilitas dengan algoritma regresi.

## Modeling
Tiga model regresi digunakan untuk memprediksi skor risiko readmisi:
1. **Regresi Linear**:
   - **Deskripsi**: Model baseline sederhana yang mengasumsikan hubungan linier antara fitur dan target.
   - **Kelebihan**: Cepat, mudah diinterpretasi.
   - **Kekurangan**: Tidak cocok untuk data dengan hubungan non-linier atau kompleks.
   - **Parameter**: Tidak ada penyetelan hiperparameter.
2. **Random Forest Regressor**:
   - **Deskripsi**: Model ensemble berbasis pohon yang menangani hubungan non-linier dan interaksi fitur.
   - **Kelebihan**: Tahan terhadap overfitting, menangani data kompleks.
   - **Kekurangan**: Komputasi intensif, sulit diinterpretasi secara langsung.
   - **Penyetelan Hiperparameter**:
     - Parameter: `n_estimators` [50, 100], `max_depth` [5, 10].
     - Metode: GridSearchCV dengan 5-fold cross-validation.
     - Hasil: Parameter terbaik meningkatkan performa (R²: 0.1064).
3. **XGBoost Regressor**:
   - **Deskripsi**: Model gradient boosting yang kuat untuk data kompleks.
   - **Kelebihan**: Performa tinggi, menangani non-linearitas dengan baik.
   - **Kekurangan**: Sensitif terhadap penyetelan, risiko overfitting tanpa regularisasi.
   - **Penyetelan Hiperparameter**:
     - Parameter: `n_estimators` [100, 200], `max_depth` [5, 7], `learning_rate` [0.1, 0.01].
     - Metode: GridSearchCV dengan 5-fold cross-validation.
     - Hasil: Parameter terbaik menghasilkan R²: 0.1103.

**Proses Improvement**:
- Penyetelan hiperparameter pada Random Forest dan XGBoost meningkatkan akurasi dibandingkan konfigurasi default.
- GridSearchCV memastikan kombinasi parameter optimal, menyeimbangkan bias dan varians.

**Pemilihan Model**:
- XGBoost dipilih sebagai model terbaik karena R² tertinggi (0.1103) dan MSE terendah (0.1098), menunjukkan kemampuan generalisasi yang lebih baik dibandingkan Regresi Linear dan Random Forest.

## Evaluation
Model dievaluasi menggunakan tiga metrik yang sesuai untuk regresi:
- **Mean Absolute Error (MAE)**: Mengukur rata-rata kesalahan absolut prediksi, memberikan gambaran akurasi secara langsung. Formula: \( MAE = \frac{1}{n} \sum |y_i - \hat{y}_i| \).
- **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat kesalahan, sensitif terhadap outlier untuk mengevaluasi kesalahan besar. Formula: \( MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \).
- **R²**: Mengukur proporsi varians data yang dijelaskan model, menunjukkan kecocokan model. Formula: \( R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} \).

**Hasil Evaluasi**:
- **Regresi Linear**:
  - MAE: 0.2928
  - MSE: 0.1136
  - R²: 0.0799
  - **Interpretasi**: Performa terendah karena asumsi linearitas tidak cocok dengan kompleksitas data.
- **Random Forest**:
  - MAE: 0.2854
  - MSE: 0.1103
  - R²: 0.1064
  - **Interpretasi**: Lebih baik dari Regresi Linear berkat kemampuan menangani non-linearitas dan penyetelan hiperparameter.
- **XGBoost**:
  - MAE: 0.2855
  - MSE: 0.1098
  - R²: 0.1103
  - **Interpretasi**: Model terbaik dengan R² tertinggi dan MSE terendah, menunjukkan generalisasi optimal meskipun R² masih rendah.

**Hubungan dengan Business Understanding**:
- **Problem Statement 1 (Biaya readmisi tinggi)**:
  - XGBoost membantu mengidentifikasi pasien berisiko tinggi, memungkinkan rumah sakit menerapkan intervensi dini untuk mengurangi biaya readmisi. Namun, R² rendah (0.1103) menunjukkan model hanya menjelaskan sebagian kecil varians, membatasi dampak penuh.
- **Problem Statement 2 (Kurang alat prediktif)**:
  - Model XGBoost menyediakan alat prediktif berbasis data, dengan pentingnya fitur seperti `number_inpatient` (0.3524) memberikan wawasan klinis tentang faktor risiko.
- **Goal 1 (Model akurat)**:
  - Tercapai sebagian; XGBoost unggul dibandingkan model lain, tetapi R² rendah menunjukkan perlunya fitur tambahan atau model lebih kuat untuk akurasi lebih tinggi.
- **Goal 2 (Wawasan klinis)**:
  - Tercapai; analisis pentingnya fitur menunjukkan `number_inpatient` dan `discharge_disposition_id` sebagai faktor utama, membantu rumah sakit fokus pada pasien dengan riwayat rawat inap.
- **Solution Statement 1 (Bandingkan tiga model)**:
  - Berhasil; XGBoost terpilih sebagai model terbaik berdasarkan metrik, memberikan dampak positif pada prediksi risiko.
- **Solution Statement 2 (Penyetelan hiperparameter)**:
  - Berhasil; penyetelan meningkatkan R² XGBoost dari konfigurasi default, meskipun peningkatan terbatas oleh kualitas fitur.

**Visualisasi**:
- Plot pentingnya fitur Random Forest disimpan sebagai `Feature.png`, menunjukkan kontribusi fitur seperti `number_inpatient` dan `discharge_disposition_id`. Gambar ini mendukung interpretasi klinis.

![Fitur Random Forest](https://github.com/MHafisAfrizal/Predictive-Analytic/blob/main/Feature.png)

## Kesimpulan
Proyek ini berhasil mengembangkan model regresi untuk memprediksi risiko readmisi pasien diabetes, dengan **XGBoost** sebagai model terbaik (R²: 0.1103, MSE: 0.1098). Model ini menjawab kebutuhan untuk alat prediktif dan memberikan wawasan klinis, meskipun R² rendah menunjukkan keterbatasan dalam menjelaskan varians data. Proyek memenuhi kriteria Dicoding, termasuk:
- Dataset kuantitatif dengan 5000 sampel.
- Dokumentasi lengkap dalam notebook dan laporan.
- Pendekatan regresi dengan tiga model dan penyetelan hiperparameter.
- Visualisasi lokal (`Feature.png`) dan analisis pentingnya fitur.

**Kelemahan**:
- R² rendah (0.08–0.11) menunjukkan model kurang kuat, mungkin karena fitur terbatas atau kompleksitas data.
- Penghapusan kolom seperti `weight` mungkin kehilangan informasi prediktif.

**Saran Perbaikan**:
- Eksplorasi fitur tambahan (misalnya, interaksi antar fitur atau data klinis baru).
- Coba algoritma lain seperti CatBoost atau teknik ensemble (stacking).
- Terapkan imputasi untuk kolom seperti `weight` daripada penghapusan.
- Perluas penyetelan hiperparameter untuk kombinasi lebih banyak.

**Dampak Bisnis**:
Model ini dapat digunakan rumah sakit untuk mengidentifikasi pasien diabetes berisiko tinggi, memungkinkan intervensi dini yang mengurangi biaya readmisi dan meningkatkan perawatan. Dengan perbaikan lebih lanjut, model dapat diintegrasikan ke sistem kesehatan untuk aplikasi dunia nyata.
