<h2>Project Title: Aspect-Based Sentiment Analysis in Indian Healthcare</h2>

<ul>
  <li>Developed an Aspect-Based Sentiment Analysis (ABSA) system to extract meaningful insights from patient reviews.</li>

  <li>Classified feedback as positive, negative, or neutral based on specific aspects such as:
    <ul>
      <li>Treatment quality</li>
      <li>Doctor behavior</li>
      <li>Staff support</li>
      <li>Medicine effectiveness</li>
    </ul>
  </li>

  <li>Used a combination of:
    <ul>
      <li>Public dataset (from open sources)</li>
      <li>Synthetic private dataset (replicated from healthcare platforms like JustDial)</li>
    </ul>
  </li>

  <li>Preprocessed the text data by:
    <ul>
      <li>Removing special characters, numbers, and stop words</li>
      <li>Converting text to lowercase</li>
      <li>Applying lemmatization for normalization</li>
    </ul>
  </li>

  <li>Extracted healthcare-related aspects using keyword-based techniques.</li>

  <li>Converted text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).</li>

  <li>Trained a Logistic Regression model to classify sentiment.</li>

  <li>Applied SMOTE (Synthetic Minority Over-sampling Technique) to handle data imbalance.</li>

  <li>Evaluated model performance using:
    <ul>
      <li>Accuracy</li>
      <li>Precision</li>
      <li>Recall</li>
      <li>F1-score</li>
    </ul>
  </li>

  <li>Visualized results using:
    <ul>
      <li>Pie charts (for sentiment distribution)</li>
      <li>ROC curves (for classification performance)</li>
    </ul>
  </li>

  <li>Achieved:
    <ul>
      <li>100% accuracy on the public dataset</li>
      <li>80% accuracy on the synthetic private dataset</li>
    </ul>
  </li>

  <li>Generated structured insights from unstructured reviews to support data-driven improvements in healthcare services using NLP and ML techniques.</li>
</ul>
