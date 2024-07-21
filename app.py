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

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1600px;
        padding: 2rem;
    }
    .dataframe-container {
        max-width: 1400px;
        margin: auto;
        overflow-x: auto;
    }
    .dataframe table {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-between;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        text-align: center;
    }
    .stImage img {
        border-radius: 15px;
        max-width: 100%;
    }
    .stButton>button {
        border-radius: 15px;
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        font-size: 16px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2C3E50;
    }
    .stMarkdown p {
        font-size: 18px;
        line-height: 1.6;
    }
    .stMarkdown .highlight {
        background-color: #FFD700;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .centered-title {
    text-align: center;
    font-size: 2rem;
    color: #2C3E50;
    font-weight: bold;
    margin: 2rem 0;
}
    </style>
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
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
    st.markdown('<div class="centered-title">Analyzing Sleep Health and Lifestyle Data</div>', unsafe_allow_html=True)
    st.subheader("Presented by Riezki Intan Pertiwi")
    st.write(
        """
        Welcome to my Streamlit application for analyzing sleep health and lifestyle data. 
        As a statistics student at Universitas Islam Indonesia, I have developed this application 
        to demonstrate the power of data analysis and machine learning in understanding and predicting sleep disorders.
        """
    )

    df = load_data()

    tabs = st.tabs(["🏠 Overview", "📊 Data Overview", "🔧 Data Preprocessing", "🚀 Model Training"])

    with tabs[0]:
        st.image("https://i.pinimg.com/564x/6c/e2/66/6ce2668a8eec2760653f88902c81f489.jpg", use_column_width=True)
        st.write("### Overview of the Analysis")
        st.write(
            """
            This application allows you to explore the relationship between various lifestyle factors and sleep disorders. 
            You can preprocess the data, train multiple machine learning models, and evaluate their performance.
            """
        )

    with tabs[1]:
        st.write("### Data Overview")
        num_rows = st.selectbox(
            "Select number of rows to display",
            options=[5, 10, 100, 'Full Data'],
            index=3,  # Default to 'Full Data'
            key='data_overview'  # Unique key for the selectbox
        )
        
        if num_rows == 'Full Data':
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(df, height=400)  # Adjust the height as needed
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(df.head(num_rows), height=400)  # Adjust the height as needed
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        if st.button('Preprocess Data'):
            df_processed = preprocessing(df)
            st.write("### Data After Preprocessing")
            st.dataframe(df_processed)  # Display all data

    with tabs[3]:
        if st.button('Train and Evaluate Models'):
            df_processed = preprocessing(df)
            results = train_and_evaluate(df_processed)
            st.write("### Model Performance")
            st.write(results)

            st.write("""
            ### Explanation of Model Performance Metrics
            The performance of the models is evaluated using the following metrics:

            - **Accuracy:** The proportion of correctly classified instances among the total instances. Higher accuracy indicates better overall performance.
            - **Precision:** The proportion of true positive results among the instances classified as positive. It measures the model's ability to avoid false positives.
            - **Recall:** The proportion of true positive results among all actual positive instances. It measures the model's ability to identify all relevant instances.
            - **F1-score:** The harmonic mean of precision and recall. It provides a balance between precision and recall and is useful when you need to balance both false positives and false negatives.

            ### Model Comparison
            Here is a summary of the model performance:

            - **Logistic Regression:** 
                - Accuracy: 88%
                - Precision: 88.7%
                - Recall: 88%
                - F1-score: 88.1%

            - **Decision Tree:**
                - Accuracy: 89.3%
                - Precision: 89.3%
                - Recall: 89.3%
                - F1-score: 89.2%

            - **Random Forest:**
                - Accuracy: 88%
                - Precision: 88.2%
                - Recall: 88%
                - F1-score: 87.9%

            From the results, the **Decision Tree** model performs slightly better compared to the **Logistic Regression** and **Random Forest** models across all metrics, indicating that it may be the best choice for this particular dataset.
            """)

            plot_results(results)

if __name__ == "__main__":
    main()
