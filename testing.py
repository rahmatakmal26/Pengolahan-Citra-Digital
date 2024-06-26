import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Memuat dataset gambar
dataset = []
labels = []

# Mengumpulkan data gambar dan label dari direktori dataset
for label in ['manggis_bagus', 'manggis_busuk']:
    path = f'Manggis/{label}'
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.resize(img, (32, 32))  # Mengubah ukuran gambar
        dataset.append(img.flatten())
        labels.append(label)


# Mengonversi label menjadi nilai numerik
le = LabelEncoder()
labels = le.fit_transform(labels)

# Memisahkan dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Mengatur jumlah tetangga terdekat
knn.fit(X_train, y_train)

# Fungsi untuk memprediksi gambar
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img_flat = img.flatten()
    prediction = knn.predict([img_flat])
    label = le.inverse_transform(prediction)[0]
    return label

# Contoh penggunaan
test_img_path = 'Citra_Testing/Citra3.jpg'
result = predict_image(test_img_path)
print(f'Gambar tersebut adalah manggis {result}')