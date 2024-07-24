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
st.set_page_config(page_title='Sleep Health and Lifestyle Analysis', layout='wide', initial_sidebar_state='expanded')

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

data = load_data()

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Set custom CSS for styling
st.markdown("""
    <style>
        .title {
            color: #5f2c82;
            font-size: 3em;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 1.5em;
            color: #49a09d;
        }
        .introduction img {
            width: 100%;
            border-radius: 10px;
        }
        .footer {
            text-align: center;
            padding: 10px;
            margin-top: 20px;
            border-top: 1px solid #ccc;
            color: #666;
        }
        .footer a {
            color: #49a09d;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# Application title
st.markdown('<h1 class="title">Sleep Health and Lifestyle Analysis</h1>', unsafe_allow_html=True)

# Sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Introduction", "Data Overview", "Visualizations", "Model Training and Performance"])

if menu == "Introduction":
    st.markdown("""
        <div class="introduction">
            <img src="https://i.pinimg.com/564x/6c/e2/66/6ce2668a8eec2760653f88902c81f489.jpg" alt="Sleep and Lifestyle">
            <h2 class="subtitle">Welcome to the Sleep Health and Lifestyle Analysis Dashboard</h2>
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
    axs[0, 0].set_title('Distribution of Sleep Duration', fontsize=16)
    axs[0, 0].set_xlabel('Duration (hours)', fontsize=14)
    axs[0, 0].set_ylabel('Frequency', fontsize=14)

    # BMI Category Count Plot
    sns.countplot(x='BMI Category', data=data, ax=axs[0, 1], palette='magma')
    axs[0, 1].set_title('BMI Category Count', fontsize=16)
    axs[0, 1].set_xlabel('BMI Category', fontsize=14)
    axs[0, 1].set_ylabel('Count', fontsize=14)

    # Stress Level Violin Plot
    sns.violinplot(x='Occupation', y='Stress Level', data=data, ax=axs[1, 0], palette='crest')
    axs[1, 0].set_title('Stress Level by Occupation', fontsize=16)
    axs[1, 0].set_xlabel('Occupation', fontsize=14)
    axs[1, 0].set_ylabel('Stress Level', fontsize=14)
    axs[1, 0].tick_params(axis='x', rotation=45)

    # Sleep Disorder Pie Chart
    data['Sleep Disorder'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), ax=axs[1, 1], startangle=140)
    axs[1, 1].set_title('Sleep Disorder Distribution', fontsize=16)

    plt.tight_layout()
    st.pyplot(fig)

    # Bar Plot Kualitas Tidur Berdasarkan Kategori BMI
    st.write("## Quality of Sleep by BMI Category")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='BMI Category', y='Quality of Sleep', data=data, palette='coolwarm')
    plt.title('Quality of Sleep by BMI Category', fontsize=16)
    plt.xlabel('BMI Category', fontsize=14)
    plt.ylabel('Quality of Sleep', fontsize=14)
    st.pyplot(plt)

elif menu == "Model Training and Performance":
    st.write("## Data Preprocessing")
    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
    data = data.drop(columns=['Blood Pressure'])
    data.dropna(subset=['Sleep Disorder'], inplace=True)
    data.fillna(data.mean(), inplace=True)
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

    # Interactive Model Performance Comparison
    st.write("## Model Performance Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=model_names, y=accuracies, name='Accuracy', marker_color='#5f2c82'))
    fig.add_trace(go.Bar(x=model_names, y=precisions, name='Precision', marker_color='#49a09d'))
    fig.add_trace(go.Bar(x=model_names, y=recalls, name='Recall', marker_color='#7b1fa2'))
    fig.add_trace(go.Bar(x=model_names, y=f1_scores, name='F1 Score', marker_color='#4caf50'))

    fig.update_layout(barmode='group', xaxis_tickangle=-45)
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
