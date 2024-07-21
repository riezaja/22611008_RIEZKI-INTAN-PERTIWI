import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_data():
    df = pd.read_csv(r'D:\UASMPMLDATA1\Sleep_health_and_lifestyle_dataset.csv')
    df.columns = df.columns.str.strip()
    return df

def preprocessing(df):
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df = df.drop(columns=['Blood Pressure'])

    label_encoder = LabelEncoder()
    for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        df[col] = label_encoder.fit_transform(df[col])

    df = df.drop(columns=['Person ID'])
    return df

def train_and_evaluate(df):
    X = df.drop('Sleep Disorder', axis=1)
    y = df['Sleep Disorder']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return results

def plot_results(results):
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    precisions = [results[model]['precision'] for model in model_names]
    recalls = [results[model]['recall'] for model in model_names]
    f1_scores = [results[model]['f1'] for model in model_names]

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

def main():
    st.title("Sleep Health and Lifestyle Analysis")

    df = load_data()
    st.write("### Data Overview")
    st.write(df.head())

    df_processed = preprocessing(df)
    st.write("### Data After Preprocessing")
    st.write(df_processed.head())

    results = train_and_evaluate(df_processed)
    st.write("### Model Performance")
    st.write(results)

    plot_results(results)

if __name__ == "__main__":
    main()
