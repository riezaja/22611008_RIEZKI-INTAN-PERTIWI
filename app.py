import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom right, #e0f7fa, #e1bee7);
        font-family: 'Courier New', Courier, monospace;
    }
    .stButton>button {
        background-color: #ff4081;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .stButton>button:hover {
        background-color: #f50057;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #673ab7;
    }
    .stMarkdown p {
        font-size: 18px;
        line-height: 1.6;
    }
    .dataframe {
        max-height: 400px;
        overflow: auto;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #673ab7;
        color: white;
        text-align: center;
        padding: 10px;
    }
    .footer p {
        margin: 0;
    }
    .header-image {
        width: 100%;
        height: auto;
        margin-bottom: 20px;
    }
    .expander-header {
        font-size: 20px;
        font-weight: bold;
        color: #673ab7;
    }
    .section-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 20px 0;
    }
    .section-container > * {
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a header image with a unique cartoon-style health-related image
st.markdown(
    """
    <div>
        <img class="header-image" src="https://i.pinimg.com/564x/6c/e2/66/6ce2668a8eec2760653f88902c81f489.jpg" alt="Health Cartoon Image">
    </div>
    """,
    unsafe_allow_html=True
)

# Add a footer with your student identity
st.markdown(
    """
    <div class="footer">
        <p>Created by Riezki Intan Pertiwi, Statistics Student at Universitas Islam Indonesia</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the dataset
def load_data():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    df.columns = df.columns.str.strip()
    return df

# Preprocess the data
def preprocessing(df):
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df = df.drop(columns=['Blood Pressure'])

    label_encoder = LabelEncoder()
    for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        df[col] = label_encoder.fit_transform(df[col])

    df = df.drop(columns=['Person ID'])
    return df

# Train and evaluate models
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

# Plot results
def plot_results(results):
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    precisions = [results[model]['precision'] for model in model_names]
    recalls = [results[model]['recall'] for model in model_names]
    f1_scores = [results[model]['f1'] for model in model_names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    sns.barplot(x=model_names, y=accuracies, ax=axes[0, 0], palette='coolwarm')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')

    sns.barplot(x=model_names, y=precisions, ax=axes[0, 1], palette='coolwarm')
    axes[0, 1].set_title('Precision')
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Score')

    sns.barplot(x=model_names, y=recalls, ax=axes[1, 0], palette='coolwarm')
    axes[1, 0].set_title('Recall')
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Score')

    sns.barplot(x=model_names, y=f1_scores, ax=axes[1, 1], palette='coolwarm')
    axes[1, 1].set_title('F1-score')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Score')

    st.pyplot(fig)

# Main function
def main():
    st.title("Sleep Health and Lifestyle Analysis")
    st.markdown("## Analyzing Sleep Health and Lifestyle Data to Understand Patterns and Predict Sleep Disorders")
    st.markdown(
        """
        This application allows you to explore the relationship between various lifestyle factors and sleep disorders. 
        You can preprocess the data, train multiple machine learning models, and evaluate their performance.
        """
    )

    df = load_data()

    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button('Show Data Overview'):
            with st.expander("Data Overview", expanded=True):
                st.dataframe(df)  # Display all data

    with col2:
        if st.button('Preprocess Data'):
            df_processed = preprocessing(df)
            with st.expander("Data After Preprocessing", expanded=True):
                st.dataframe(df_processed)  # Display all data

                results = train_and_evaluate(df_processed)
                st.write("### Model Performance")
                st.write(results)

                st.write("### Performance Metrics Visualization")
                plot_results(results)

if __name__ == "__main__":
    main()
