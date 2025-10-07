import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Membuat Dataset Contoh
# Anda harus mengganti ini dengan dataset Anda sendiri.
# Kolom 'gaji_tahunan' adalah contoh kriteria.
# Kolom 'penerima_bansos' adalah label (0 = Tidak, 1 = Ya).
data = {
    'gaji_per_bulan': [2500000, 3000000, 1800000, 4500000, 2000000, 3500000, 1500000, 5000000, 2200000, 2800000],
    'jumlah_tanggungan': [2, 1, 3, 0, 4, 2, 5, 1, 3, 2],
    'kepemilikan_rumah': ['sewa', 'milik_pribadi', 'sewa', 'milik_pribadi', 'sewa', 'milik_pribadi', 'sewa', 'milik_pribadi', 'sewa', 'milik_pribadi'],
    'penerima_bansos': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Definisikan UMR (contoh, sesuaikan dengan UMR di daerah Anda)
umr = 3000000

# Tambahkan fitur biner baru: 'di_bawah_umr'
# 1 jika gaji di bawah UMR, 0 jika tidak
df['di_bawah_umr'] = (df['gaji_per_bulan'] < umr).astype(int)

# 2. Pra-pemrosesan Data (Mengubah data kategorikal menjadi numerik)
# Gunakan One-Hot Encoding untuk kolom 'kepemilikan_rumah'
df = pd.get_dummies(df, columns=['kepemilikan_rumah'], drop_first=True)

# 3. Memisahkan Fitur dan Label
# Fitur (X) adalah kolom yang akan digunakan untuk memprediksi
# Label (y) adalah kolom target (penerima_bansos)
X = df[['di_bawah_umr', 'jumlah_tanggungan', 'kepemilikan_rumah_sewa']]
y = df['penerima_bansos']

# 4. Membagi Data Menjadi Data Latih dan Data Uji
# Data latih digunakan untuk melatih model
# Data uji digunakan untuk menguji performa model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Membangun Model Naive Bayes (Gaussian Naive Bayes)
# Model ini cocok untuk fitur numerik yang diasumsikan berdistribusi Gaussian
gnb = GaussianNB()

# 6. Melatih Model
gnb.fit(X_train, y_train)

# 7. Melakukan Prediksi
y_pred = gnb.predict(X_test)

# 8. Mengevaluasi Performa Model
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))
print("\nMatriks Kebingungan:")
print(confusion_matrix(y_test, y_pred))

# 9. Contoh Prediksi untuk Data Baru
# Buat data baru yang ingin Anda prediksi
data_baru = {
    'gaji_per_bulan': [2100000, 4000000],
    'jumlah_tanggungan': [3, 1],
    'kepemilikan_rumah': ['sewa', 'milik_pribadi']
}
df_baru = pd.DataFrame(data_baru)

# Pra-pemrosesan data baru sama seperti data latih
df_baru['di_bawah_umr'] = (df_baru['gaji_per_bulan'] < umr).astype(int)
# Buat kolom dummy yang sama
df_baru = pd.get_dummies(df_baru, columns=['kepemilikan_rumah'], drop_first=True)

# Pastikan kolom data baru sama persis dengan kolom data latih
# Jika 'kepemilikan_rumah_sewa' tidak ada, tambahkan dengan nilai 0
if 'kepemilikan_rumah_sewa' not in df_baru.columns:
    df_baru['kepemilikan_rumah_sewa'] = 0

# Urutkan kolom agar sesuai dengan model
X_baru = df_baru[['di_bawah_umr', 'jumlah_tanggungan', 'kepemilikan_rumah_sewa']]

# Lakukan prediksi
prediksi_baru = gnb.predict(X_baru)
prediksi_prob = gnb.predict_proba(X_baru)

# Tampilkan hasil prediksi
print("\n--- Hasil Prediksi Data Baru ---")
for i, pred in enumerate(prediksi_baru):
    status = "Penerima Bansos" if pred == 1 else "Bukan Penerima Bansos"
    print(f"Orang ke-{i+1} dengan gaji {data_baru['gaji_per_bulan'][i]} dan tanggungan {data_baru['jumlah_tanggungan'][i]}: {status}")
    print(f"Probabilitas: Tidak Menerima = {prediksi_prob[i][0]:.2f}, Menerima = {prediksi_prob[i][1]:.2f}")