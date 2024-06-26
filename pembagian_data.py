import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Fungsi untuk melakukan pembagian, pelatihan model, dan menampilkan hasilnya
def split_train_evaluate(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Inisialisasi dan latih model (menggunakan Random Forest sebagai contoh)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    
    print(f"Pembagian {100-test_size*100:.0f}:{test_size*100:.0f}")
    print(f"Jumlah data latih: {len(X_train)}")
    print(f"Jumlah data uji: {len(X_test)}")
    print(f"Akurasi: {accuracy:.2f}%")
    print(f"Presisi: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-score: {f1:.2f}%")
    print("--------------------")

# Membaca file CSV
try:
    df = pd.read_csv('dataset.csv')
    print("Dataset berhasil dibaca.")
    print(f"Jumlah total data: {len(df)}")
    print("--------------------")

    # Asumsikan kolom terakhir adalah target/label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Melakukan pembagian, pelatihan, dan evaluasi untuk berbagai rasio
    split_train_evaluate(X, y, 0.2)  # 80:20
    split_train_evaluate(X, y, 0.3)  # 70:30
    split_train_evaluate(X, y, 0.4)  # 60:40

except FileNotFoundError:
    print("File 'dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan script ini.")
except pd.errors.EmptyDataError:
    print("File 'dataset.csv' kosong.")
except Exception as e:
    print(f"Terjadi kesalahan: {str(e)}")