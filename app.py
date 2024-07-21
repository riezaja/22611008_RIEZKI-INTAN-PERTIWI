import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

# Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }

    model_dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model_dt, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

# Cross-validation
def cross_validation(X, y):
    model_dt = DecisionTreeClassifier(random_state=42)
    cv_scores = cross_val_score(model_dt, X, y, cv=5, scoring='accuracy')
    return cv_scores

# Voting Classifier
def voting_classifier(X_train, y_train):
    model_lr = LogisticRegression(random_state=42)
    model_dt = DecisionTreeClassifier(random_state=42)
    model_rf = RandomForestClassifier(random_state=42)

    voting_clf = VotingClassifier(estimators=[
        ('lr', model_lr),
        ('dt', model_dt),
        ('rf', model_rf)
    ], voting='soft')

    voting_clf.fit(X_train, y_train)
    return voting_clf

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

    tabs = st.tabs([
        "🏠 Overview", "📊 Data Overview", "📉 Visualisasi Durasi Tidur", 
        "🔧 Data Preprocessing", "🚀 Model Training and Comparison", 
        "📈 Bar Plots", "🌡️ Heatmap", "🔍 Hyperparameter Tuning", 
        "🔄 Cross-validation", "🗳️ Model Voting Classifier"
    ])

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
            index=3,
            key='data_overview'
        )
        
        if num_rows == 'Full Data':
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(df, height=400)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(df.head(num_rows), height=400)
            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.write("### Visualisasi Distribusi Durasi Tidur")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Sleep Duration'], bins=30, kde=True, color='skyblue')
        plt.title('Distribusi Durasi Tidur')
        plt.xlabel('Durasi Tidur (jam)')
        plt.ylabel('Frekuensi')
        st.pyplot()

    with tabs[3]:
        st.write("### Data Preprocessing")
        st.write("Before preprocessing:")
        st.write(df.head())

        df_preprocessed = preprocessing(df)
        st.write("After preprocessing:")
        st.write(df_preprocessed.head())

    with tabs[4]:
        st.write("### Model Training and Comparison")
        results = train_and_evaluate(df_preprocessed)
        st.write(results)
        plot_results(results)

    with tabs[5]:
        st.write("### Bar Plots")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Gender', hue='Sleep Disorder', palette='coolwarm')
        plt.title('Sleep Disorder Distribution by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        st.pyplot()

    with tabs[6]:
        st.write("### Heatmap")
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix Heatmap')
        st.pyplot()

    with tabs[7]:
        st.write("### Hyperparameter Tuning")
        X = df_preprocessed.drop('Sleep Disorder', axis=1)
        y = df_preprocessed['Sleep Disorder']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_estimator, best_params, best_score = hyperparameter_tuning(X_train, y_train)
        st.write("Best Estimator:")
        st.write(best_estimator)
        st.write("Best Parameters:")
        st.write(best_params)
        st.write("Best Score:")
        st.write(best_score)

    with tabs[8]:
        st.write("### Cross-validation")
        X = df_preprocessed.drop('Sleep Disorder', axis=1)
        y = df_preprocessed['Sleep Disorder']
        
        cv_scores = cross_validation(X, y)
        st.write("Cross-validation scores:")
        st.write(cv_scores)
        st.write("Mean CV Score:", cv_scores.mean())

    with tabs[9]:
        st.write("### Model Voting Classifier")
        X = df_preprocessed.drop('Sleep Disorder', axis=1)
        y = df_preprocessed['Sleep Disorder']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        voting_clf = voting_classifier(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        
        st.write("Voting Classifier Results:")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
        st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

if __name__ == '__main__':
    main()
