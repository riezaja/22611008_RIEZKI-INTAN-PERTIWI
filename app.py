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
    background: radial-gradient(circle, #f9a9b1, #d95dae); /* Radial gradient for a dynamic background */
    color: #333;
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}

.main .block-container {
    max-width: 900px; /* Reduced width for a more compact container */
    padding: 1.5rem;
    background: rgba(255, 255, 255, 0.9); /* Semi-transparent background */
    border-radius: 20px; /* Smaller rounded corners */
    box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2); /* Lighter shadow for depth */
    margin: 1.5rem auto;
    border: 1px solid rgba(0, 0, 0, 0.1); /* Subtle border for definition */
    position: relative;
}

.centered-title {
    text-align: center;
    font-size: 2.5rem; /* Smaller font size for a more compact title */
    color: #fff;
    font-weight: 700;
    margin: 1.5rem 0;
    background: linear-gradient(to right, #e0aaff, #a76cd9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.4); /* Subtle shadow for impact */
}
.stButton>button {
    border-radius: 20px; /* Smaller rounded corners */
    background-color: #a76cd9;
    color: white;
    padding: 0.8rem 2rem; /* Reduced padding */
    font-size: 16px; /* Smaller font size */
    margin-top: 1rem;
    transition: background-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
    font-family: 'Arial', sans-serif;
    border: none;
    cursor: pointer;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); /* Lighter shadow */
}

.stButton>button:hover {
    background-color: #8a4d8d;
    transform: scale(1.03);
    box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3); /* Slightly stronger shadow on hover */
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #5f2c82;
    font-family: 'Georgia', serif;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3); /* Subtle shadow for headings */
}

.stMarkdown p {
    font-size: 16px; /* Smaller font size */
    line-height: 1.5;
    margin-bottom: 1rem; /* Reduced margin */
}

.dataframe-container {
    display: flex;
    justify-content: center;
    margin: 1rem 0; /* Adjusted margin */
}

.tab-content {
    padding: 1.5rem;
    background: rgba(255, 255, 255, 0.9); /* Semi-transparent background */
    border-radius: 20px; /* Smaller rounded corners */
    box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2); /* Medium shadow for depth */
    position: relative;
}

.tab-content h3 {
    color: #5f2c82;
    font-family: 'Georgia', serif;
    font-size: 1.8rem; /* Smaller font size */
    border-bottom: 2px solid #a76cd9; /* Thinner underline */
    padding-bottom: 0.4rem;
    margin-bottom: 1rem; /* Reduced margin */
    position: relative;
}

.tab-content h3::before {
    content: '';
    position: absolute;
    left: 50%;
    bottom: -6px; /* Position slightly below the text */
    transform: translateX(-50%);
    width: 30px;
    height: 3px;
    background-color: #a76cd9;
    border-radius: 3px;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2); /* Subtle shadow for the underline */
}

.tab-content p {
    font-size: 14px; /* Smaller font size */
    line-height: 1.4;
}

.tab-content img {
    display: block;
    margin: 1rem auto;
    border-radius: 15px; /* Smaller border-radius */
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); /* Light shadow effect */
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.tab-content img:hover {
    transform: scale(1.05); /* Slight zoom effect on hover */
    box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3); /* Slightly stronger shadow on hover */
}

.tabs {
    display: flex;
    justify-content: center;
    margin-bottom: 1rem; /* Adjusted margin */
}

.tab-button {
    padding: 0.6rem 1.2rem; /* Smaller padding */
    margin: 0 0.3rem;
    border: 1px solid #a76cd9;
    border-radius: 20px; /* Rounded corners */
    background-color: #ffffff;
    color: #a76cd9;
    font-size: 14px; /* Smaller font size */
    cursor: pointer;
    transition: background-color 0.3s ease, color 0.3s ease, transform 0.3s ease;
    font-family: 'Arial', sans-serif;
    box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);
}

.tab-button:hover {
    background-color: #a76cd9;
    color: #ffffff;
    transform: scale(1.02);
}

.active {
    background-color: #a76cd9;
    color: #ffffff;
}

.footer {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(to right, #f9a9b1, #d95dae);
    color: #ffffff;
    font-size: 14px; /* Smaller font size */
    border-top: 1px solid rgba(255, 255, 255, 0.2); /* Subtle border for definition */
    margin-top: 1.5rem;
    position: relative;
}

.footer::before {
    display: block;
    font-size: 12px; /* Smaller font size for footer text */
    color: #ffffff;
    margin-top: 0.5rem;
}

.footer a {
    color: #ffffff;
    text-decoration: none;
    font-weight: bold;
    transition: color 0.3s ease;
}

.footer a:hover {
    color: #f9a9b1; /* Highlight link on hover */
    text-decoration: underline;
}

/* Icons */
.icon {
    font-size: 1rem; /* Smaller icon size */
    color: #a76cd9;
    margin-right: 0.3rem;
    vertical-align: middle;
}

/* Example of Font Awesome usage */
.icon-heart:before {
    content: "\f004"; /* Unicode for heart icon */
    font-family: 'Font Awesome 5 Free';
    font-weight: 900;
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

    # Display the data before preprocessing
    st.write("### Data Before Preprocessing")
    st.dataframe(data)

    # Define preprocessing function
    def preprocessing(df):
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df = df.drop(columns=['Blood Pressure'])

        label_encoder = LabelEncoder()
        for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
            df[col] = label_encoder.fit_transform(df[col])

        df = df.drop(columns=['Person ID'])
        return df

    # Apply preprocessing
    preprocessed_data = preprocessing(data.copy())

    # Display the data after preprocessing
    st.write("### Data After Preprocessing")
    st.dataframe(preprocessed_data)

    st.write("Missing Values per Column:")
    st.write(preprocessed_data.isnull().sum())

    # Fill missing values only for numeric columns
    numeric_cols = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns
    preprocessed_data[numeric_cols] = preprocessed_data[numeric_cols].fillna(preprocessed_data[numeric_cols].mean())

    preprocessed_data.dropna(subset=['Sleep Disorder'], inplace=True)

    X = preprocessed_data.drop('Sleep Disorder', axis=1)
    y = preprocessed_data['Sleep Disorder']
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
