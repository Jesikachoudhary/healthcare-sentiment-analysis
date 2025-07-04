import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Public Dataset
public_data = pd.read_csv('publicdataset.tsv', sep='\t')  # Replace with the correct file path

# Preprocessing the Public Dataset
X_public = public_data['Feedback']
y_public = public_data['Sentiment']  # Assuming 'Sentiment' column exists

# Train-test split for Public Dataset
X_train_public, X_test_public, y_train_public, y_test_public = train_test_split(X_public, y_public, test_size=0.2, random_state=42)

# TF-IDF Vectorizer for Public Dataset
vectorizer_public = TfidfVectorizer(stop_words='english')
X_train_public_tfidf = vectorizer_public.fit_transform(X_train_public)
X_test_public_tfidf = vectorizer_public.transform(X_test_public)

# Use SMOTE to handle class imbalance for Public Dataset
smote_public = SMOTE(random_state=42, k_neighbors=3)  # Using k_neighbors=3 to avoid the error
X_train_public_smote, y_train_public_smote = smote_public.fit_resample(X_train_public_tfidf, y_train_public)

# Model: Logistic Regression for Public Dataset
model_public = LogisticRegression()
model_public.fit(X_train_public_smote, y_train_public_smote)

# Predictions for Public Dataset
y_pred_public = model_public.predict(X_test_public_tfidf)

# Calculate Accuracy for Public Dataset
accuracy_public = accuracy_score(y_test_public, y_pred_public)
print(f'Accuracy of Public Dataset model: {accuracy_public:.2f}')


# Load the Private Dataset
private_data = pd.read_csv('privatedataset.tsv', sep='\t')  # Replace with the correct file path

# Preprocessing the Private Dataset
X_private = private_data['Feedback']
y_private = private_data['Sentiment']

# Train-test split for Private Dataset
X_train_private, X_test_private, y_train_private, y_test_private = train_test_split(X_private, y_private, test_size=0.2, random_state=42)

# TF-IDF Vectorizer for Private Dataset
vectorizer_private = TfidfVectorizer(stop_words='english')
X_train_private_tfidf = vectorizer_private.fit_transform(X_train_private)
X_test_private_tfidf = vectorizer_private.transform(X_test_private)

# Use SMOTE to handle class imbalance for Private Dataset
smote_private = SMOTE(random_state=42, k_neighbors=3)  # Using k_neighbors=3 to avoid the error
X_train_private_smote, y_train_private_smote = smote_private.fit_resample(X_train_private_tfidf, y_train_private)

# Model: Logistic Regression for Private Dataset
model_private = LogisticRegression()
model_private.fit(X_train_private_smote, y_train_private_smote)

# Predictions for Private Dataset
y_pred_private = model_private.predict(X_test_private_tfidf)

# Calculate Accuracy for Private Dataset
accuracy_private = accuracy_score(y_test_private, y_pred_private)
print(f'Accuracy of Private Dataset model: {accuracy_private:.2f}')


# Plotting the accuracy for both datasets
labels = ['Public Dataset', 'Private Dataset']
accuracies = [accuracy_public, accuracy_private]

plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison between Public and Private Datasets')
plt.ylim(0, 1)  # Accuracy is between 0 and 1
plt.show()

