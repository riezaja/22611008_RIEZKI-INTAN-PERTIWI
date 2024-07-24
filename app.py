import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import plotly.graph_objects as go

# Set up the layout and page configuration
st.set_page_config(page_title='Sleep Health and Lifestyle Analysis', layout='wide')

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

data = load_data()

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Add custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: 'Times New Roman', serif;
        background: linear-gradient(to right, #5f2c82, #49a09d); /* Gradient background in purple and teal */
    }
    .main .block-container {
        max-width: 1200px;
        padding: 2rem;
        background: linear-gradient(to right, #ffffff, #f0f0f0); /* Light gradient background for the container */
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
    .centered-title {
        text-align: center;
        font-size: 2.8rem;
        color: #333; /* Dark gray for the title text */
        font-weight: 700;
        margin: 2rem 0;
        background: linear-gradient(to right, #5f2c82, #49a09d); /* Gradient text background */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        border-radius: 12px;
        background-color: #5f2c82; /* Purple button */
        color: white;
        padding: 0.8rem 2rem;
        font-size: 18px;
        margin-top: 1rem;
        transition: background-color 0.3s ease;
        font-family: 'Times New Roman', serif;
    }
    .stButton>button:hover {
        background-color: #4e1f66; /* Slightly darker purple on hover */
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #333; /* Dark gray text */
        font-family: 'Times New Roman', serif;
    }
    .stMarkdown p {
        font-size: 18px;
        line-height: 1.8;
        margin-bottom: 1.2rem;
    }
    .dataframe-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .tab-content {
        padding: 2rem;
        background: linear-gradient(to right, #ffffff, #f0f0f0); /* Light gradient background for tabs */
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .tab-content h3 {
        color: #333; /* Dark gray text for tab headers */
    }
    .tab-content p {
        font-size: 16px;
        line-height: 1.6;
    }
    .tab-content img {
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Application title
st.markdown('<h1 class="centered-title">Sleep Health and Lifestyle Analysis</h1>', unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Introduction", "Data Overview", "Visualizations", "Preprocessing, Model Training, and Model Performance"])

if menu == "Introduction":
    st.markdown("""
        <div class="tab-content">
            <img src="https://i.pinimg.com/564x/6c/e2/66/6ce2668a8eec2760653f88902c81f489.jpg" alt="Sleep and Lifestyle">
            <h2>Welcome to the Sleep Health and Lifestyle Analysis Dashboard</h2>
            <p>This application provides insights into sleep health and lifestyle factors using a comprehensive dataset. Explore the data overview, visualizations, and machine learning models to understand how different factors affect sleep quality and overall health.</p>
        </div>
    """, unsafe_allow_html=True)

elif menu == "Data Overview":
    st.write("## Data Overview")
    st.dataframe(data)

    st.write("## Summary Statistics")
    st.write(data.describe())

elif menu == "Visualizations":
    st.write("## Enhanced Visualizations")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Sleep Duration Distribution
    sns.set(style="whitegrid", palette="pastel")
    sns.histplot(data['Sleep Duration'], bins=30, ax=axs[0, 0], kde=True, color='#5f2c82')
    axs[0, 0].set_title('Distribusi Durasi Tidur', fontsize=16)
    axs[0, 0].set_xlabel('Durasi Tidur (jam)', fontsize=14)
    axs[0, 0].set_ylabel('Frekuensi', fontsize=14)

    # BMI Category Count Plot
    sns.countplot(x='BMI Category', data=data, ax=axs[0, 1], palette='magma')
    axs[0, 1].set_title('Count Plot of BMI Category', fontsize=16)
    axs[0, 1].set_xlabel('Kategori BMI', fontsize=14)
    axs[0, 1].set_ylabel('Frekuensi', fontsize=14)

    # Stress Level Violin Plot
    sns.violinplot(x='Occupation', y='Stress Level', data=data, ax=axs[1, 0], palette='crest')
    axs[1, 0].set_title('Violin Plot of Stress Level by Occupation', fontsize=16)
    axs[1, 0].set_xlabel('Pekerjaan', fontsize=14)
    axs[1, 0].set_ylabel('Tingkat Stres', fontsize=14)
    axs[1, 0].tick_params(axis='x', rotation=45)

    # Sleep Disorder Pie Chart
    data['Sleep Disorder'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), ax=axs[1, 1], startangle=140)
    axs[1, 1].set_title('Distribusi Kategori Gangguan Tidur', fontsize=16)

    plt.tight_layout()
    st.pyplot(fig)

    # Bar Plot Kualitas Tidur Berdasarkan Kategori BMI
    plt.figure(figsize=(10, 6))
    sns.barplot(x='BMI Category', y='Quality of Sleep', data=data, palette='coolwarm')
    plt.title('Bar Plot Kualitas Tidur Berdasarkan Kategori BMI', fontsize=16)
    plt.xlabel('Kategori BMI', fontsize=14)
    plt.ylabel('Kualitas Tidur', fontsize=14)
    st.pyplot(plt)

elif menu == "Preprocessing, Model Training, and Model Performance":
    st.write("## Preprocessing Data")
    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
    data = data.drop(columns=['Blood Pressure'])
    st.write("Missing Values per Column:")
    st.write(data.isnull().sum())
    
    # Fill missing values only for numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    data.dropna(subset=['Sleep Disorder'], inplace=True)
    label_encoder = LabelEncoder()
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])
    data = data.drop(columns=['Person ID'])

    X = data.drop('Sleep Disorder', axis=1)
    y = data['Sleep Disorder']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training and evaluation
    st.write("## Model Performance")
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy, precision, recall, f1

    model_lr = LogisticRegression(random_state=42)
    model_dt = DecisionTreeClassifier(random_state=42)
    model_rf = RandomForestClassifier(random_state=42)
    
    # Fit models
    model_lr.fit(X_train, y_train)
    model_dt.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)

    # Evaluate models
    st.write("### Logistic Regression")
    lr_acc, lr_prec, lr_rec, lr_f1 = evaluate_model(model_lr, X_test, y_test)
    st.write(f"Accuracy: {lr_acc:.4f}, Precision: {lr_prec:.4f}, Recall: {lr_rec:.4f}, F1-Score: {lr_f1:.4f}")

    st.write("### Decision Tree")
    dt_acc, dt_prec, dt_rec, dt_f1 = evaluate_model(model_dt, X_test, y_test)
    st.write(f"Accuracy: {dt_acc:.4f}, Precision: {dt_prec:.4f}, Recall: {dt_rec:.4f}, F1-Score: {dt_f1:.4f}")

    st.write("### Random Forest")
    rf_acc, rf_prec, rf_rec, rf_f1 = evaluate_model(model_rf, X_test, y_test)
    st.write(f"Accuracy: {rf_acc:.4f}, Precision: {rf_prec:.4f}, Recall: {rf_rec:.4f}, F1-Score: {rf_f1:.4f}")

    # Hyperparameter tuning for Decision Tree
    st.write("## Hyperparameter Tuning for Decision Tree")
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    st.write("Best Parameters:", grid_search.best_params_)
    st.write("Best Score:", grid_search.best_score_)

    # Cross-validation
    st.write("## Cross-Validation Results")
    cv_scores = cross_val_score(model_rf, X, y, cv=5, scoring='accuracy')
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Mean CV Score: {cv_scores.mean():.4f}")

    # Voting Classifier
    st.write("## Voting Classifier")
    voting_clf = VotingClassifier(estimators=[
        ('lr', model_lr),
        ('dt', model_dt),
        ('rf', model_rf)
    ], voting='hard')
    voting_clf.fit(X_train, y_train)
    voting_acc, voting_prec, voting_rec, voting_f1 = evaluate_model(voting_clf, X_test, y_test)
    st.write(f"**Voting Classifier:** Accuracy={voting_acc:.4f}, Precision={voting_prec:.4f}, Recall={voting_rec:.4f}, F1-Score={voting_f1:.4f}")

    # Custom Footer
    st.markdown("""
        <div class="footer">
            Created with ❤️ by Riezki Intan Pertiwi | <a href="https://www.uii.ac.id" style="color: #49a09d;">Universitas Islam Indonesia</a>
        </div>
    """, unsafe_allow_html=True)
