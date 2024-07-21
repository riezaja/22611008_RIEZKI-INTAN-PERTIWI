import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
df = pd.read_csv(r'D:\UASMPMLDATA1\Sleep_health_and_lifestyle_dataset.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Menampilkan informasi dasar
print(df.info())
print(df.describe())
print(df.head())

# Visualisasi distribusi durasi tidur
plt.figure(figsize=(10, 6))
sns.histplot(df['Sleep Duration'], bins=30)
plt.title('Distribusi Durasi Tidur')
plt.xlabel('Durasi Tidur (jam)')
plt.ylabel('Frekuensi')
plt.show()
