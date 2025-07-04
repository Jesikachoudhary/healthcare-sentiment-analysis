import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Example data for public dataset (replace with actual values)
y_test_pub = [1, 0, 1, 1, 0, 1, 0, 1]  # True labels for public dataset
y_pred_pub = [1, 0, 1, 0, 0, 1, 0, 1]  # Predicted labels for public dataset

# Example data for private dataset (replace with actual values)
y_test_private = [1, 1, 0, 1, 0, 0, 1, 1]  # True labels for private dataset
y_pred_private = [1, 1, 0, 0, 0, 1, 1, 1]  # Predicted labels for private dataset

# Function to calculate confusion matrix and plot pie chart
def plot_pie_chart(true_labels, predicted_labels, dataset_name):
    # Get confusion matrix values
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Labels and sizes for the pie chart
    labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
    sizes = [tp, tn, fp, fn]
    colors = ['#4CAF50', '#2196F3', '#FF5722', '#FFC107']  # Colors for each category

    # Create pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(f'Classification Performance for {dataset_name} Dataset')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# Plot pie charts for both datasets
plot_pie_chart(y_test_pub, y_pred_pub, 'Public')
plot_pie_chart(y_test_private, y_pred_private, 'Private')

