import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    data.columns = data.columns.str.strip()
    return data

data = load_data()

# Sidebar options
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", ['Data Overview', 'Data Preprocessing', 'Model Training and Evaluation', 'Model Performance'])

# Data Overview
if options == 'Data Overview':
    st.title("Data Overview")
    st.write(data.head())
    st.write(data.describe())
    st.write(data.info())

    # Visualizations
    st.subheader("Distribusi Durasi Tidur")
    fig, ax = plt.subplots()
    sns.histplot(data['Sleep Duration'], bins=30, ax=ax)
    ax.set_title('Distribusi Durasi Tidur')
    ax.set_xlabel('Durasi Tidur (jam)')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    st.subheader("Box Plot of Sleep Duration by Gender")
    fig, ax = plt.subplots()
    sns.boxplot(x='Gender', y='Sleep Duration', data=data, ax=ax)
    ax.set_title('Box Plot of Sleep Duration by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Durasi Tidur (jam)')
    st.pyplot(fig)

    st.subheader("Count Plot of BMI Category")
    fig, ax = plt.subplots()
    sns.countplot(x='BMI Category', data=data, ax=ax)
    ax.set_title('Count Plot of BMI Category')
    ax.set_xlabel('Kategori BMI')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    st.subheader("Violin Plot of Stress Level by Occupation")
    fig, ax = plt.subplots()
    sns.violinplot(x='Occupation', y='Stress Level', data=data, ax=ax)
    ax.set_title('Violin Plot of Stress Level by Occupation')
    ax.set_xlabel('Pekerjaan')
    ax.set_ylabel('Tingkat Stres')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Data Preprocessing
elif options == 'Data Preprocessing':
    st.title("Data Preprocessing")

    # Handle missing values (if any)
    st.write("Missing values per column:")
    st.write(data.isnull().sum())

    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        data[col] = label_encoder.fit_transform(data[col])

    # Split Blood Pressure into Systolic and Diastolic
    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
    data = data.drop(columns=['Blood Pressure', 'Person ID'])

    st.write("Data after preprocessing:")
    st.write(data.head())
    st.write(data.dtypes)

# Model Training and Evaluation
elif options == 'Model Training and Evaluation':
    st.title("Model Training and Evaluation")

    # Define features and target
    X = data.drop('Sleep Disorder', axis=1)
    y = data['Sleep Disorder']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define evaluate_model function
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        return accuracy, precision, recall, f1, report

    # Train models
    model_lr = LogisticRegression(random_state=42)
    model_dt = DecisionTreeClassifier(random_state=42)
    model_rf = RandomForestClassifier(random_state=42)

    models = {'Logistic Regression': model_lr, 'Decision Tree': model_dt, 'Random Forest': model_rf}

    for name, model in models.items():
        model.fit(X_train, y_train)

    # Model evaluation
    st.subheader("Model Performance")
    results = {}
    for name, model in models.items():
        accuracy, precision, recall, f1, report = evaluate_model(model, X_test, y_test)
        results[name] = [accuracy, precision, recall, f1]
        st.write(f"{name}:\n")
        st.text(report)

    # Model comparison visualization
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'F1 Score']).T
    st.dataframe(results_df)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', ax=ax)
    ax.set_title('Model Comparison')
    st.pyplot(fig)

# Model Performance
elif options == 'Model Performance':
    st.title("Model Performance and Hyperparameter Tuning")

    # Cross-validation
    model_dt = DecisionTreeClassifier(random_state=42)
    cv_scores = cross_val_score(model_dt, X, y, cv=5, scoring='accuracy')
    st.write("Cross-Validation Scores:", cv_scores)
    st.write("Mean Cross-Validation Score:", cv_scores.mean())

    # Hyperparameter tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    grid_search = GridSearchCV(estimator=model_dt, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    st.write("Best Parameters:", grid_search.best_params_)
    st.write("Best Score:", grid_search.best_score_)

    # Voting Classifier
    model_lr = LogisticRegression(random_state=42)
    model_dt_best = grid_search.best_estimator_
    model_rf = RandomForestClassifier(random_state=42)
    voting_clf = VotingClassifier(estimators=[('lr', model_lr), ('dt', model_dt_best), ('rf', model_rf)], voting='soft')
    voting_clf.fit(X_train, y_train)

    acc_vc, prec_vc, rec_vc, f1_vc, report_vc = evaluate_model(voting_clf, X_test, y_test)
    st.write("Voting Classifier Performance:\n")
    st.text(report_vc)
