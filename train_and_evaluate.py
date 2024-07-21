#train_and_evaluate.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#c. Untuk tugas klasifikasi, gunakan metrik seperti akurasi, presisi, perolehan, dan skor F1
# Define evaluate_model function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    return accuracy, precision, recall, f1

# Read data
data = pd.read_csv(r'D:\UASMPMLDATA1\Sleep_health_and_lifestyle_dataset.csv')

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Split Blood Pressure into Systolic and Diastolic
data[['Systolic_BP', 'Diastolic_BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
data = data.drop(columns=['Blood Pressure'])

# Encoding labels
label_encoder = LabelEncoder()
for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    data[col] = label_encoder.fit_transform(data[col])

# Drop unnecessary columns
data = data.drop(columns=['Person ID'])

# Define features and target
X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
model_lr = LogisticRegression(random_state=42)
model_dt = DecisionTreeClassifier(random_state=42)
model_rf = RandomForestClassifier(random_state=42)

#f. Tulis kode Python untuk melatih dan mengevaluasi setiap model, termasuk penyetelan hyperparameter yang diperlukan.
# Fit models
model_lr.fit(X_train, y_train)
model_dt.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Evaluate models using evaluate_model function
print("Logistic Regression:")
acc_lr, prec_lr, rec_lr, f1_lr = evaluate_model(model_lr, X_test, y_test)
print("\nDecision Tree:")
acc_dt, prec_dt, rec_dt, f1_dt = evaluate_model(model_dt, X_test, y_test)
print("\nRandom Forest:")
acc_rf, prec_rf, rec_rf, f1_rf = evaluate_model(model_rf, X_test, y_test)

#e. Berikan visualisasi untuk membandingkan kinerja model
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
plt.show()


# Heatmap
performance_matrix = np.array([accuracies, precisions, recalls, f1_scores])
performance_df = pd.DataFrame(performance_matrix, index=['Accuracy', 'Precision', 'Recall', 'F1-score'], columns=model_names)
plt.figure(figsize=(10, 6))
sns.heatmap(performance_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('Model Performance Heatmap')
plt.show()

from sklearn.model_selection import GridSearchCV

# Define parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}

# Initialize RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Use best estimator to evaluate on the test set
best_model_rf = grid_search.best_estimator_
acc_rf, prec_rf, rec_rf, f1_rf = evaluate_model(best_model_rf, X_test, y_test)

from sklearn.model_selection import cross_val_score

# Initialize RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model_rf, X, y, cv=5, scoring='accuracy')

# Print cross-validation results
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())
