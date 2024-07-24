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

# Set custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        .title {
            font-family: 'Roboto', sans-serif;
            color: #5f2c82;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-family: 'Roboto', sans-serif;
            color: #49a09d;
        }
        .container {
            display: grid;
            grid-template-columns: 1fr 3fr;
            gap: 20px;
            padding: 20px;
        }
        .sidebar {
            background-color: #f2f2f2;
            padding: 20px;
            border-radius: 10px;
        }
        .section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            font-family: 'Roboto', sans-serif;
            font-size: 24px;
            color: #5f2c82;
            margin-bottom: 10px;
        }
        .footer {
            font-family: 'Roboto', sans-serif;
            color: #49a09d;
            text-align: center;
            padding: 10px;
            border-top: 1px solid #e0e0e0;
        }
        .chart-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Application title
st.markdown('<h1 class="title">Sleep Health and Lifestyle Analysis</h1>', unsafe_allow_html=True)

# Layout for the main content
with st.container():
    st.write("## Data Overview")
    st.dataframe(data)

    st.write("## Summary Statistics")
    st.write(data.describe())

    # Visualization section
    st.write("## Enhanced Visualizations")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Sleep Duration Distribution
    sns.histplot(data['Sleep Duration'], bins=30, ax=axs[0, 0], kde=True, color='purple')
    axs[0, 0].set_title('Distribusi Durasi Tidur')
    axs[0, 0].set_xlabel('Durasi Tidur (jam)')
    axs[0, 0].set_ylabel('Frekuensi')

    # BMI Category Count Plot
    sns.countplot(x='BMI Category', data=data, ax=axs[0, 1], palette='magma')
    axs[0, 1].set_title('Count Plot of BMI Category')
    axs[0, 1].set_xlabel('Kategori BMI')
    axs[0, 1].set_ylabel('Frekuensi')

    # Stress Level Violin Plot
    sns.violinplot(x='Occupation', y='Stress Level', data=data, ax=axs[1, 0], palette='crest')
    axs[1, 0].set_title('Violin Plot of Stress Level by Occupation')
    axs[1, 0].set_xlabel('Pekerjaan')
    axs[1, 0].set_ylabel('Tingkat Stres')
    axs[1, 0].tick_params(axis='x', rotation=45)

    # Sleep Disorder Pie Chart
    data['Sleep Disorder'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), ax=axs[1, 1], startangle=140)
    axs[1, 1].set_title('Distribusi Kategori Gangguan Tidur')

    plt.tight_layout()
    st.pyplot(fig)

    # Bar Plot Kualitas Tidur Berdasarkan Kategori BMI
    plt.figure(figsize=(10, 6))
    sns.barplot(x='BMI Category', y='Quality of Sleep', data=data, palette='coolwarm')
    plt.title('Bar Plot Kualitas Tidur Berdasarkan Kategori BMI')
    plt.xlabel('Kategori BMI')
    plt.ylabel('Kualitas Tidur')
    st.pyplot(plt)

    # Preprocessing steps
    st.write("## Preprocessing Data")
    data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
    data = data.drop(columns=['Blood Pressure'])
    st.write("Missing Values per Column:")
    st.write(data.isnull().sum())
    data.dropna(subset=['Sleep Disorder'], inplace=True)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    label_encoder = LabelEncoder()
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])
    data = data.drop(columns=['Person ID'])

    X = data.drop('Sleep Disorder', axis=1)
    y = data['Sleep Disorder']
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
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
st.markdown('<div class="footer">Created with ❤️ by Riezki Intan Pertiwi | Universitas Islam Indonesia</div>', unsafe_allow_html=True)
