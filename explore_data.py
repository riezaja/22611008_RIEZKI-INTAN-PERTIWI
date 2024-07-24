#explore_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca dataset
df = pd.read_csv(r'D:\UASMPMLDATA\Sleep_health_and_lifestyle_dataset.csv')

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


# Count plot for BMI Category
plt.figure(figsize=(8, 6))
sns.countplot(x='BMI Category', data=df)
plt.title('Count Plot of BMI Category')
plt.xlabel('Kategori BMI')
plt.ylabel('Frekuensi')
plt.show()

# Violin plot for Stress Level by Occupation
plt.figure(figsize=(10, 6))
sns.violinplot(x='Occupation', y='Stress Level', data=df)
plt.title('Violin Plot of Stress Level by Occupation')
plt.xlabel('Pekerjaan')
plt.ylabel('Tingkat Stres')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.show()

# Pie Chart Kategori Gangguan Tidur
plt.figure(figsize=(8, 8))
df['Sleep Disorder'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Distribusi Kategori Gangguan Tidur')
plt.savefig('pie_chart_sleep_disorder.png')  # Simpan gambar ke file
plt.show()  # Tampilkan gambar

# Bar Plot Kualitas Tidur Berdasarkan Kategori BMI
plt.figure(figsize=(10, 6))
sns.barplot(x='BMI Category', y='Quality of Sleep', data=df)
plt.title('Bar Plot Kualitas Tidur Berdasarkan Kategori BMI')
plt.xlabel('Kategori BMI')
plt.ylabel('Kualitas Tidur')
plt.savefig('bar_plot_quality_of_sleep.png')  # Simpan gambar ke file
plt.show()  # Tampilkan gambar

