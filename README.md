# Healthcare Sentiment Analysis

This project implements Aspect-Based Sentiment Analysis (ABSA) on Indian healthcare reviews to classify sentiments related to specific aspects such as treatment quality, doctor behavior, staff support, and medicine effectiveness.

The objective is to extract insights from healthcare service reviews using machine learning and NLP techniques to assist healthcare providers in improving service quality.

---

## Features

- Aspect tagging using keyword-based extraction
- Sentiment classification using TF-IDF and Logistic Regression
- Handling of class imbalance with SMOTE
- Visualization using ROC curves and pie charts
- Compatible with both public and synthetic private datasets

---

## Project Structure

- `sentiment.py`: Core ML model for sentiment analysis
- `publicdataset.py`: Processing and modeling on public dataset
- `privatedataset.py`: Processing and modeling on synthetic dataset
- `accuracy.check.py`: Model evaluation and accuracy reporting
- `complete.py`: Integrated end-to-end pipeline
- `ROCcurve.py`: Generates ROC curve for classifier performance
- `piechart.py`: Pie chart for aspect sentiment distribution
- `Train and predict.ipynb`: Jupyter notebook for model testing

---

## Datasets

- **Public Dataset**: Cleaned and manually labeled review dataset
- **Private Dataset**: Synthetic dataset created with support from Jio Platforms, used for academic research purposes

---

## Results

- Achieved 100% accuracy on a small public dataset
- Achieved approximately 80% accuracy on the synthetic private dataset
- Evaluation includes confusion matrix, accuracy score, and ROC curves

---

