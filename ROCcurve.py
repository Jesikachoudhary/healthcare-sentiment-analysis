import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Load datasets
public_df = pd.read_csv('publicdataset.tsv', sep='\t')
private_df = pd.read_csv('privatedataset.tsv', sep='\t')

# Clean column names (strip spaces and ensure correct casing)
public_df.columns = public_df.columns.str.strip()
private_df.columns = private_df.columns.str.strip()

# Print the columns to confirm their names
print("Public dataset columns:", public_df.columns.tolist())
print("Private dataset columns:", private_df.columns.tolist())

# Prepare data function
def prepare_data(df, text_col, label_col):
    X = df[text_col]
    y = df[label_col]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorizer (converts text to numeric data)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Public dataset preparation
X_train_pub, X_test_pub, y_train_pub, y_test_pub = prepare_data(public_df, 'Feedback', 'Sentiment')

# TF-IDF transformation
X_train_pub_vec = vectorizer.fit_transform(X_train_pub.astype(str))
X_test_pub_vec = vectorizer.transform(X_test_pub.astype(str))

# Train the model for the public dataset
model_pub = LogisticRegression()
model_pub.fit(X_train_pub_vec, y_train_pub)
y_scores_pub = model_pub.predict_proba(X_test_pub_vec)[:, 1]

# Private dataset preparation
X_train_priv, X_test_priv, y_train_priv, y_test_priv = prepare_data(private_df, 'Feedback', 'Sentiment')

# TF-IDF transformation
X_train_priv_vec = vectorizer.fit_transform(X_train_priv.astype(str))
X_test_priv_vec = vectorizer.transform(X_test_priv.astype(str))

# Train the model for the private dataset
model_priv = LogisticRegression()
model_priv.fit(X_train_priv_vec, y_train_priv)
y_scores_priv = model_priv.predict_proba(X_test_priv_vec)[:, 1]

# Compute ROC curve for both datasets
fpr_pub, tpr_pub, _ = roc_curve(y_test_pub, y_scores_pub)
fpr_priv, tpr_priv, _ = roc_curve(y_test_priv, y_scores_priv)

roc_auc_pub = auc(fpr_pub, tpr_pub)
roc_auc_priv = auc(fpr_priv, tpr_priv)

# Plot ROC curve for both datasets
plt.figure(figsize=(8, 6))
plt.plot(fpr_pub, tpr_pub, color='blue', label=f'Public Dataset (AUC = {roc_auc_pub:.2f})')
plt.plot(fpr_priv, tpr_priv, color='green', label=f'Private Dataset (AUC = {roc_auc_priv:.2f})')
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

