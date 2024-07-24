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
    font-family: 'Arial', sans-serif;
    background: radial-gradient(circle, #f9a9b1, #d95dae);
    color: #333;
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}

.main .block-container {
    max-width: 1000px; /* Adjusted width for better fit */
    padding: 2rem; /* Reduced padding for a tighter layout */
    background: rgba(255, 255, 255, 0.95);
    border-radius: 30px; /* Slightly rounded corners */
    box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.4);
    margin: 2rem auto;
    border: 1px solid rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
}

.centered-title {
    text-align: center;
    font-size: 3.5rem; /* Reduced font size for better fit */
    color: #fff;
    font-weight: 900;
    margin: 2rem 0;
    background: linear-gradient(to right, #e0aaff, #a76cd9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 5px 5px 10px rgba(0, 0, 0, 0.7);
    animation: pulse 3s infinite;
}

@keyframes pulse {
    0% {
        text-shadow: 5px 5px 10px rgba(0, 0, 0, 0.7);
    }
    50% {
        text-shadow: 8px 8px 15px rgba(0, 0, 0, 0.5);
    }
    100% {
        text-shadow: 5px 5px 10px rgba(0, 0, 0, 0.7);
    }
}

.stButton>button {
    border-radius: 30px; /* Rounded corners */
    background: linear-gradient(to right, #a76cd9, #a44c77);
    color: #fff;
    padding: 1.25rem 3rem; /* Adjusted padding */
    font-size: 22px; /* Slightly smaller font size */
    margin-top: 1rem;
    transition: background 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
    font-family: 'Arial', sans-serif;
    border: none;
    cursor: pointer;
    box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.3);
}

.stButton>button:hover {
    background: linear-gradient(to right, #8a4d8d, #7a3b6d);
    transform: scale(1.05);
    box-shadow: 0px 12px 24px rgba(0, 0, 0, 0.4);
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #5f2c82;
    font-family: 'Georgia', serif;
    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    letter-spacing: 0.5px;
}

.stMarkdown p {
    font-size: 18px; /* Adjusted font size for readability */
    line-height: 1.7;
    margin-bottom: 1.5rem;
}

.dataframe-container {
    display: flex;
    justify-content: center;
    margin: 1.5rem 0;
}

.tab-content {
    padding: 2rem;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 30px;
    box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.tab-content h3 {
    color: #5f2c82;
    font-family: 'Georgia', serif;
    font-size: 2.2rem; /* Reduced font size */
    border-bottom: 4px solid #a76cd9;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.2);
}

.tab-content h3::before {
    content: '';
    position: absolute;
    left: 50%;
    bottom: -10px;
    transform: translateX(-50%);
    width: 60px;
    height: 4px;
    background-color: #a76cd9;
    border-radius: 10px;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
}

.tab-content p {
    font-size: 16px; /* Smaller font size */
    line-height: 1.6;
}

.tab-content img {
    display: block;
    margin: 1.5rem auto;
    border-radius: 20px; /* Moderate border-radius */
    box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.3);
    transition: transform 0.4s ease, box-shadow 0.4s ease;
}

.tab-content img:hover {
    transform: scale(1.05);
    box-shadow: 0px 12px 30px rgba(0, 0, 0, 0.4);
}

.tabs {
    display: flex;
    justify-content: center;
    margin-bottom: 1.5rem;
}

.tab-button {
    padding: 1rem 2rem;
    margin: 0 0.5rem;
    border: 1px solid #a76cd9;
    border-radius: 30px; /* Rounded corners */
    background: #ffffff;
    color: #a76cd9;
    font-size: 18px; /* Adjusted font size */
    cursor: pointer;
    transition: background 0.3s ease, color 0.3s ease, transform 0.3s ease;
    font-family: 'Arial', sans-serif;
    box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.15);
}

.tab-button:hover {
    background: #a76cd9;
    color: #ffffff;
    transform: scale(1.05);
}

.active {
    background: #a76cd9;
    color: #ffffff;
}

.footer {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(to right, #f9a9b1, #d95dae);
    color: #ffffff;
    font-size: 18px; /* Adjusted font size */
    border-top: 2px solid rgba(255, 255, 255, 0.2);
    margin-top: 2rem;
    position: relative;
    box-shadow: 0px -10px 20px rgba(0, 0, 0, 0.3);
}

.footer::before {
    content: 'Created with ♥ by Riezki Intan Pertiwi';
    display: block;
    font-size: 16px; /* Adjusted font size */
    color: #ffffff;
    margin-top: 0.5rem;
}

.footer a {
    color: #ffffff;
    text-decoration: none;
    font-weight: bold;
    transition: color 0.3s ease, text-shadow 0.3s ease;
}

.footer a:hover {
    color: #f9a9b1;
    text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.4);
    text-decoration: underline;
}

</style>


    """,
    unsafe_allow_html=True
)

# Application title
st.markdown('<h1 class="centered-title">Sleep Health and Lifestyle Analysis</h1>', unsafe_allow_html=True)

# Horizontal tab menu
tabs = ["Introduction", "Data Overview", "Visualizations", "Preprocessing, Model Training, and Model Performance"]
tab = st.selectbox("Choose a tab", tabs, index=0, key="tabs")

if tab == "Introduction":
    st.markdown("""
        <div class="tab-content">
            <img src="https://i.pinimg.com/564x/6c/e2/66/6ce2668a8eec2760653f88902c81f489.jpg" alt="Sleep and Lifestyle" width="600">
            <h2>Welcome to the Sleep Health and Lifestyle Analysis Dashboard</h2>
            <p>This application provides insights into sleep health and lifestyle factors using a comprehensive dataset. Explore the data overview, visualizations, and machine learning models to understand how different factors affect sleep quality and overall health.</p>
        </div>
    """, unsafe_allow_html=True)

elif tab == "Data Overview":
    st.write("## Data Overview")
    st.dataframe(data)

    st.write("## Summary Statistics")
    st.write(data.describe())

elif tab == "Visualizations":
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

elif tab == "Preprocessing, Model Training, and Model Performance":
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

    model_lr.fit(X_train, y_train)
    model_dt.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)

    acc_lr, prec_lr, rec_lr, f1_lr = evaluate_model(model_lr, X_test, y_test)
    acc_dt, prec_dt, rec_dt, f1_dt = evaluate_model(model_dt, X_test, y_test)
    acc_rf, prec_rf, rec_rf, f1_rf = evaluate_model(model_rf, X_test, y_test)

    st.write(f"**Logistic Regression:** Accuracy={acc_lr:.4f}, Precision={prec_lr:.4f}, Recall={rec_lr:.4f}, F1-Score={f1_lr:.4f}")
    st.write(f"**Decision Tree:** Accuracy={acc_dt:.4f}, Precision={prec_dt:.4f}, Recall={rec_dt:.4f}, F1-Score={f1_dt:.4f}")
    st.write(f"**Random Forest:** Accuracy={acc_rf:.4f}, Precision={prec_rf:.4f}, Recall={rec_rf:.4f}, F1-Score={f1_rf:.4f}")

    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    accuracies = [acc_lr, acc_dt, acc_rf]
    precisions = [prec_lr, prec_dt, prec_rf]
    recalls = [rec_lr, rec_dt, rec_rf]
    f1_scores = [f1_lr, f1_dt, f1_rf]

    st.write("## Model Performance Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_names, y=accuracies, name='Accuracy'))
    fig.add_trace(go.Bar(x=model_names, y=precisions, name='Precision'))
    fig.add_trace(go.Bar(x=model_names, y=recalls, name='Recall'))
    fig.add_trace(go.Bar(x=model_names, y=f1_scores, name='F1-Score'))

    fig.update_layout(barmode='group', title='Model Performance Metrics', xaxis_title='Model', yaxis_title='Score')
    st.plotly_chart(fig)

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
