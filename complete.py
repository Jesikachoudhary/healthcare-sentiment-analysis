import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Sample text for tokenization
sample_text = "The doctor was very kind and the treatment worked effectively!"

# Step 1: Tokenization
tokens = word_tokenize(sample_text)

# Step 2: Convert tokens to lowercase
tokens = [token.lower() for token in tokens]

# Step 3: Remove punctuation and special characters
tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]

# Step 4: Remove empty tokens
tokens = [token for token in tokens if token]

# Step 5: Get the set of stop words
stop_words = set(stopwords.words('english'))

# Step 6: Remove stop words
cleaned_tokens = [token for token in tokens if token not in stop_words]

# Print cleaned tokens
print("Cleaned Tokens:", cleaned_tokens)

# Step 7: Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Step 8: Lemmatize the cleaned tokens
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]

# Print lemmatized tokens
print("Lemmatized Tokens:", lemmatized_tokens)

# Optional: Frequency Distribution
from nltk import FreqDist

# Create a frequency distribution of the lemmatized tokens
freq_dist = FreqDist(lemmatized_tokens)

# Print the most common tokens
print("Most Common Tokens:", freq_dist.most_common(5))
