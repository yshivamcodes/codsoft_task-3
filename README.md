# Credit Card Fraud Detection

## Project Overview:
This project implements a machine learning model to detect fraudulent credit card transactions using the Credit Card Fraud Detection dataset. The model employs a Random Forest Classifier with undersampling to address class imbalance, achieving high recall for fraud detection. The dataset is processed and analyzed in a Jupyter Notebook, including exploratory data analysis (EDA), preprocessing, model training, and evaluation.

## Dataset
The Credit Card Fraud Detection dataset (creditcard.csv) contains 284,807 transactions with the following features:
Time: Seconds elapsed between each transaction and the first transaction.
V1 to V28: Anonymized features resulting from PCA transformation.
Amount: Transaction amount.
Class: Target variable (0 = Non-Fraud, 1 = Fraud).

The dataset is sourced from a local file path in the provided notebook and is highly imbalanced, with only 0.17% of transactions labeled as fraudulent.

## Project Structure
CREDIT CARD FRAUD DETECTION.ipynb: Jupyter Notebook containing the complete code for data loading, EDA, preprocessing, handling class imbalance, model training, and evaluation.
creditcard.csv: The dataset file (not included in this repository; ensure it is available in your local environment or update the file path in the notebook).

## Requirements
To run the notebook, install the following Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

## Methodology
### Data Loading:
The dataset is loaded using pandas from the specified file path.

### Exploratory Data Analysis (EDA):
Analyzed dataset shape (284,807 rows, 31 columns), data types, missing values (none found), and descriptive statistics.

### Visualizations include:
Distribution of Class (highly imbalanced: 0 = Non-Fraud, 1 = Fraud).
Histogram of scaled Amount and Time features.
Correlation heatmap of all features.
Correlation of features with Class to identify influential features (e.g., V11, V4, V2 have positive correlations; V17, V14, V12 have strong negative correlations).

### Data Preprocessing:
Scaled Amount and Time features using StandardScaler.
Split data into features (X: all columns except Class) and target (y: Class).
Performed train-test split (80% train, 20% test) with stratification to maintain class distribution.

### Handling Class Imbalance:
Used undersampling to balance the training set by downsampling the majority class (non-fraud) to match the minority class (fraud) using resample from scikit-learn.
Resulting balanced training set contains equal numbers of fraud and non-fraud samples.

### Model Training:
Trained a Random Forest Classifier (random_state=42) on the balanced training data.

### Evaluation:
Evaluated the model on the test set using classification_report.

### Key metrics:
=== Random Forest ===
            precision    recall  f1-score   support
       0       1.00      0.96      0.98     56864
       1       0.04      0.92      0.08        98
accuracy                           0.96     56962

macro avg 0.52 0.94 0.53 56962 weighted avg 1.00 0.96 0.98 56962

* - High recall (0.92) for the fraud class (1) indicates effective detection of fraudulent transactions, though precision (0.04) is low due to class imbalance in the test set. *

## Results

### Model Performance:
Accuracy: 96% (overall, dominated by the majority class).
Recall for Fraud (Class 1): 92%, indicating the model successfully identifies most fraudulent transactions.
Precision for Fraud (Class 1): 4%, reflecting a high number of false positives due to the severe class imbalance.
F1-Score for Fraud (Class 1): 8%, balancing precision and recall.
The model prioritizes high recall for fraud detection, which is critical for minimizing missed fraudulent transactions in real-world applications.
