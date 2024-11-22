import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

# Load the dataset
data = pd.read_csv("HeartDisease/heart.csv")

# Label Encoding for binary categorical variables (like 'sex', 'fbs', etc.)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['fbs'] = label_encoder.fit_transform(data['fbs'])
data['exang'] = label_encoder.fit_transform(data['exang'])

# One-Hot Encoding for multi-class categorical variables
data = pd.get_dummies(data, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

scaler = StandardScaler()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Univariate distribution of age and cholesterol
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data['age'], kde=True, bins=20)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
sns.histplot(data['chol'], kde=True, bins=20)
plt.title('Cholesterol Distribution')

plt.tight_layout()
plt.savefig('age_cholesterol_distribution.png')

# Correlation matrix
corr = data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)


svm_classifier = SVC(probability=True)
svm_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred_svm = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred_gb = gb_classifier.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)

# Generate confusion matrices for the models
cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title('Logistic Regression')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Random Forest')

sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('SVM')

sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('Gradient Boosting')

plt.tight_layout()
plt.savefig('confusion_matrices.png')

# Function to plot ROC Curve
def plot_roc_curve(fpr, tpr, label):
    plt.plot(fpr, tpr, linewidth=2, label=label)

# Logistic Regression ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
auc_lr = auc(fpr_lr, tpr_lr)

# Random Forest ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
auc_rf = auc(fpr_rf, tpr_rf)

# SVM ROC Curve
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_classifier.predict_proba(X_test)[:, 1])
auc_svm = auc(fpr_svm, tpr_svm)

# Gradient Boosting ROC Curve
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_classifier.predict_proba(X_test)[:, 1])
auc_gb = auc(fpr_gb, tpr_gb)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr_lr, tpr_lr, f'Logistic Regression (AUC = {auc_lr:.2f})')
plot_roc_curve(fpr_rf, tpr_rf, f'Random Forest (AUC = {auc_rf:.2f})')
plot_roc_curve(fpr_svm, tpr_svm, f'SVM (AUC = {auc_svm:.2f})')
plot_roc_curve(fpr_gb, tpr_gb, f'Gradient Boosting (AUC = {auc_gb:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig('roc_curve_comparison.png')

# Feature importance for Random Forest
features = X.columns
importances = rf_classifier.feature_importances_

# Plotting the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance (Random Forest)')
plt.savefig('feature_importance.png')
