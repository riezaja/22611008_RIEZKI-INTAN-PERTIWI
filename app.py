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

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

data = load_data()

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a section:", ["Data Overview", "Visualizations", "Model Training", "Hyperparameter Tuning", "Cross-validation", "Voting Classifier"])

# Display data overview
if options == "Data Overview":
    st.title('Sleep Health and Lifestyle Analysis')
    st.write("## Data Overview")
    st.dataframe(data)
    st.write("## Summary Statistics")
    st.write(data.describe())

elif options == "Visualizations":
    st.title('Data Visualizations')
    st.write("## Visualize Data")

    # Visualize Sleep Duration distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Sleep Duration'], bins=30)
    plt.title('Distribusi Durasi Tidur')
    plt.xlabel('Durasi Tidur (jam)')
    plt.ylabel('Frekuensi')
    st.pyplot(plt)

    # Add more visualizations as needed
    # For example, visualizing correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# Preprocessing steps
st.write("## Preprocessing Data")
# Split Blood Pressure into Systolic and Diastolic
data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
data = data.drop(columns=['Blood Pressure'])

# Check for missing values
st.write("Missing Values per Column:")
st.write(data.isnull().sum())

# Drop rows with missing target values
data.dropna(subset=['Sleep Disorder'], inplace=True)

# Fill missing numeric values with the mean
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Drop unnecessary columns
data = data.drop(columns=['Person ID'])

# Define features and target
X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  # Replace any remaining NaNs with 0 or another value if necessary

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if options == "Model Training":
    st.write("## Model Training and Evaluation")

    # Initialize models
    model_lr = LogisticRegression(random_state=42)
    model_dt = DecisionTreeClassifier(random_state=42)
    model_rf = RandomForestClassifier(random_state=42)

    # Train and evaluate models
    model_lr.fit(X_train, y_train)
    model_dt.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy, precision, recall, f1

    acc_lr, prec_lr, rec_lr, f1_lr = evaluate_model(model_lr, X_test, y_test)
    acc_dt, prec_dt, rec_dt, f1_dt = evaluate_model(model_dt, X_test, y_test)
    acc_rf, prec_rf, rec_rf, f1_rf = evaluate_model(model_rf, X_test, y_test)

    # Display evaluation results
    st.write(f"**Logistic Regression:** Accuracy={acc_lr:.4f}, Precision={prec_lr:.4f}, Recall={rec_lr:.4f}, F1-Score={f1_lr:.4f}")
    st.write(f"**Decision Tree:** Accuracy={acc_dt:.4f}, Precision={prec_dt:.4f}, Recall={rec_dt:.4f}, F1-Score={f1_dt:.4f}")
    st.write(f"**Random Forest:** Accuracy={acc_rf:.4f}, Precision={prec_rf:.4f}, Recall={rec_rf:.4f}, F1-Score={f1_rf:.4f}")

    # Visualization of model comparisons
    model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    accuracies = [acc_lr, acc_dt, acc_rf]
    precisions = [prec_lr, prec_dt, prec_rf]
    recalls = [rec_lr, rec_dt, rec_rf]
    f1_scores = [f1_lr, f1_dt, f1_rf]

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

if options == "Hyperparameter Tuning":
    st.write("## Hyperparameter Tuning for Decision Tree")
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    grid_search = GridSearchCV(estimator=model_dt, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    st.write(f"Best Parameters: {best_params}")
    best_model_dt = grid_search.best_estimator_
    acc_dt, prec_dt, rec_dt, f1_dt = evaluate_model(best_model_dt, X_test, y_test)
    st.write(f"**Best Decision Tree:** Accuracy={acc_dt:.4f}, Precision={prec_dt:.4f}, Recall={rec_dt:.4f}, F1-Score={f1_dt:.4f}")

if options == "Cross-validation":
    st.write("## Cross-validation")
    cv_scores = cross_val_score(model_dt, X, y, cv=5, scoring='accuracy')
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Mean Cross-Validation Score: {cv_scores.mean():.4f}")

if options == "Voting Classifier":
    st.write("## Voting Classifier")
    voting_clf = VotingClassifier(estimators=[
        ('lr', model_lr),
        ('dt', model_dt),
        ('rf', model_rf)
    ], voting='soft')
    voting_clf.fit(X_train, y_train)
    acc_vc, prec_vc, rec_vc, f1_vc = evaluate_model(voting_clf, X_test, y_test)
    st.write(f"**Voting Classifier:** Accuracy={acc_vc:.4f}, Precision={prec_vc:.4f}, Recall={rec_vc:.4f}, F1-Score={f1_vc:.4f}")
