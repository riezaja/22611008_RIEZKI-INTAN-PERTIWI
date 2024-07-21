#preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv(r'D:\UASMPMLDATA\Sleep_health_and_lifestyle_dataset.csv')

# Periksa nilai hilang
print("Nilai hilang per kolom:")
print(data.isnull().sum())

# Tangani nilai hilang (ganti dengan strategi yang sesuai)
# Misalnya, mengisi kolom numerik dengan rata-rata atau median
# data['Sleep Duration (hours)'].fillna(data['Sleep Duration (hours)'].mean(), inplace=True)
# Untuk kolom kategorik, pertimbangkan imputasi modus atau membuat kategori baru
# Buat objek LabelEncoder
label_encoder = LabelEncoder()
for col in ['Person ID', 'Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    data[col] = label_encoder.fit_transform(data[col])

# Kodekan kolom kategorik
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Occupation'] = label_encoder.fit_transform(data['Occupation'])
data['BMI Category'] = label_encoder.fit_transform(data['BMI Category'])
data['Sleep Disorder'] = label_encoder.fit_transform(data['Sleep Disorder'])

# Buat objek StandardScaler
scaler = StandardScaler()

# Skala fitur numerik (jika diperlukan)
# data['Sleep Duration (hours)'] = scaler.fit_transform(data[['Sleep Duration (hours)']])
print("\nTipe data setelah preprocessing:")
print(data.dtypes)
