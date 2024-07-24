import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1, classification_report(y_test, y_pred)

# Load and preprocess data
def load_data():
    data = pd.read_csv('D:\\UASMPMLDATA\\Sleep_health_and_lifestyle_dataset.csv')
    data.columns = data.columns.str.strip()

    # Handle missing values and encode categorical variables
    label_encoder = LabelEncoder()
    for col in ['Person ID', 'Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        data[col] = label_encoder.fit_transform(data[col])

    # Split Blood Pressure into Systolic and Diastolic
    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
    data = data.drop(columns=['Blood Pressure', 'Person ID'])
    
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.title('Sleep Health Analysis')
option = st.sidebar.radio('Navigation', ['Data Overview', 'EDA', 'Model Training & Evaluation'])

if option == 'Data Overview':
    st.title('Data Overview')
    st.write('#### Dataset Information')
    st.write(data.info())
    st.write('#### First Five Rows of the Data')
    st.write(data.head())

elif option == 'EDA':
    st.title('Exploratory Data Analysis')

    st.subheader('Distribusi Durasi Tidur')
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Sleep Duration'], bins=30)
    plt.title('Distribusi Durasi Tidur')
    plt.xlabel('Durasi Tidur (jam)')
    plt.ylabel('Frekuensi')
    st.pyplot(plt)

    st.subheader('Box Plot of Sleep Duration by Gender')
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Gender', y='Sleep Duration', data=data)
    plt.title('Box Plot of Sleep Duration by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Durasi Tidur (jam)')
    st.pyplot(plt)

    st.subheader('Count Plot of BMI Category')
    plt.figure(figsize=(8, 6))
    sns.countplot(x='BMI Category', data=data)
    plt.title('Count Plot of BMI Category')
    plt.xlabel('Kategori BMI')
    plt.ylabel('Frekuensi')
    st.pyplot(plt)

    st.subheader('Violin Plot of Stress Level by Occupation')
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Occupation', y='Stress Level', data=data)
    plt.title('Violin Plot of Stress Level by Occupation')
    plt.xlabel('Pekerjaan')
    plt.ylabel('Tingkat Stres')
    plt.xticks(rotation=45)
    st.pyplot(plt)

elif option == 'Model Training & Evaluation':
    st.title('Model Training & Evaluation')
    
    # Define features and target
    X = data.drop('Sleep Disorder', axis=1)
    y = data['Sleep Disorder']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize models
    model_lr = LogisticRegression(random_state=42)
    model_dt = DecisionTreeClassifier(random_state=42)
    model_rf = RandomForestClassifier(random_state=42)

    # Fit models
    model_lr.fit(X_train, y_train)
    model_dt.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)

    # Evaluate models
    acc_lr, prec_lr, rec_lr, f1_lr, report_lr = evaluate_model(model_lr, X_test, y_test)
    acc_dt, prec_dt, rec_dt, f1_dt, report_dt = evaluate_model(model_dt, X_test, y_test)
    acc_rf, prec_rf, rec_rf, f1_rf, report_rf = evaluate_model(model_rf, X_test, y_test)

    st.subheader('Logistic Regression')
    st.text(report_lr)
    st.subheader('Decision Tree')
    st.text(report_dt)
    st.subheader('Random Forest')
    st.text(report_rf)

    # Visualization of model comparisons
    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    accuracies = [acc_lr, acc_dt, acc_rf]
    precisions = [prec_lr, prec_dt, prec_rf]
    recalls = [rec_lr, rec_dt, rec_rf]
    f1_scores = [f1_lr, f1_dt, f1_rf]

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    sns.barplot(x=model_names, y=accuracies)
    plt.title('Accuracy')

    plt.subplot(2, 2, 2)
    sns.barplot(x=model_names, y=precisions)
    plt.title('Precision')

    plt.subplot(2, 2, 3)
    sns.barplot(x=model_names, y=recalls)
    plt.title('Recall')

    plt.subplot(2, 2, 4)
    sns.barplot(x=model_names, y=f1_scores)
    plt.title('F1-score')

    plt.tight_layout()
    st.pyplot(plt)
