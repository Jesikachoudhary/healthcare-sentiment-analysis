import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")
# Step 1: Load the dataset (make sure the correct path to your file is provided)
file_path = "privatedataset.tsv"
public_df = pd.read_csv(file_path, sep='\t')
# Step 2: Check the data structure
print("Columns in dataset:", public_df.columns)
print(public_df.head())
# Step 3: Preprocessing the dataset
# Ensure there are no missing values in the columns we need
public_df = public_df.dropna(subset=["Feedback", "Sentiment"])
# Step 4: Split the dataset into features (X) and target (y)
X = public_df['Feedback']
y = public_df['Sentiment']
# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 6: Convert text data into numerical data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Step 7: Handle class imbalance using SMOTE (if necessary)
# Check class distribution before SMOTE
print("Class distribution in y_train before SMOTE:")
print(y_train.value_counts())
# Adjust n_neighbors to be smaller or equal to the minority class size
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)  # Changed k_neighbors to 2
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
# Check class distribution after SMOTE
print("Class distribution in y_train after SMOTE:")
print(pd.Series(y_train_smote).value_counts())
# Step 8: Train a Logistic Regression model with class weights to handle imbalance
# Convert the classes list to a numpy array
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train_smote)
class_weights = {0: class_weights[0], 1: class_weights[1]}
# Initialize Logistic Regression model with class weights
model = LogisticRegression(class_weight=class_weights, max_iter=1000)
# Step 9: Fit the model on the resampled training data
model.fit(X_train_smote, y_train_smote)
# Step 10: Make predictions on the test set
y_pred = model.predict(X_test_tfidf)
# Step 11: Evaluate the model using classification report and accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
# Step 12: Cross-validation (optional, if you want to check model performance on multiple folds)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='accuracy')
print("Cross-validated accuracy:", cv_scores.mean(), "+/-", cv_scores.std())
