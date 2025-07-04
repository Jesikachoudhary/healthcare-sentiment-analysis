from textblob import TextBlob
import nltk
nltk.download('punkt')

# Sample patient review
review = "The doctor was very kind and the treatment worked effectively!"

# Create a TextBlob object
blob = TextBlob(review)

# Analyze sentiment
sentiment = blob.sentiment

# Output
print(f"Review: {review}")
print(f"Polarity: {sentiment.polarity}")    # Range: [-1.0, 1.0]
print(f"Subjectivity: {sentiment.subjectivity}")  # Range: [0.0, 1.0]

# Interpretation
if sentiment.polarity > 0:
    print("Sentiment: Positive")
elif sentiment.polarity < 0:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")
