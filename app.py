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

# Filtering numerical columns for correlation heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Pair plot for selected numerical features
numerical_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
sns.pairplot(df[numerical_features], diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.show()

# Box plot for Sleep Duration by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Sleep Duration', data=df)
plt.title('Box Plot of Sleep Duration by Gender')
plt.xlabel('Gender')
plt.ylabel('Durasi Tidur (jam)')
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

# Additional plots based on specific variables of interest
# Example: Box plot for Quality of Sleep by Sleep Disorder status
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sleep Disorder', y='Quality of Sleep', data=df)
plt.title('Box Plot of Quality of Sleep by Sleep Disorder')
plt.xlabel('Gangguan Tidur')
plt.ylabel('Kualitas Tidur')
plt.show()
