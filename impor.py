import os
import pandas as pd
from PIL import Image
import numpy as np
from scipy.stats import skew, kurtosis  # Impor fungsi skew dan kurtosis dari scipy.stats

# Fungsi untuk mengekstrak fitur citra
def extract_features(image_path, class_label):
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Menghitung mean, std, skewness, dan kurtosis untuk setiap channel warna
    mean_r = np.mean(image_array[:,:,0])
    mean_g = np.mean(image_array[:,:,1])
    mean_b = np.mean(image_array[:,:,2])
    
    std_r = np.std(image_array[:,:,0])
    std_g = np.std(image_array[:,:,1])
    std_b = np.std(image_array[:,:,2])
    
    skew_r = skew(image_array[:,:,0].flatten())
    skew_g = skew(image_array[:,:,1].flatten())
    skew_b = skew(image_array[:,:,2].flatten())
    
    kurt_r = kurtosis(image_array[:,:,0].flatten())  # Menggunakan fungsi kurtosis dari scipy.stats
    kurt_g = kurtosis(image_array[:,:,1].flatten())
    kurt_b = kurtosis(image_array[:,:,2].flatten())
    
    features = [mean_r, mean_g, mean_b, std_r, std_g, std_b, skew_r, skew_g, skew_b, kurt_r, kurt_g, kurt_b, class_label]
    return features
# Folder pertama
folder1 = 'Manggis/manggis_busuk'
class_label1 = 'manggis_busuk'

# Folder kedua
folder2 = 'Manggis/manggis_bagus'
class_label2 = 'manggis_bagus'

# Daftar untuk menyimpan fitur
features_list = []

# Memproses folder pertama
for filename in os.listdir(folder1):
    image_path = os.path.join(folder1, filename)
    features = extract_features(image_path, class_label1)
    features_list.append(features)

# Memproses folder kedua
for filename in os.listdir(folder2):
    image_path = os.path.join(folder2, filename)
    features = extract_features(image_path, class_label2)
    features_list.append(features)

# Membuat DataFrame dari daftar fitur
column_names = ['meanR', 'meanG', 'meanB', 'stdR', 'stdG', 'stdB', 'skewR', 'skewG', 'skewB', 'kurtR', 'kurtG', 'kurtB', 'kelas']
dataset = pd.DataFrame(features_list, columns=column_names)

# Menyimpan dataset ke file CSV
dataset.to_csv('dataset.csv', index=False)