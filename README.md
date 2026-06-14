# NLP Spam Classifier

## Project Overview

This project builds a machine learning model to classify SMS messages as either Spam or Ham (non-spam) using Natural Language Processing (NLP) techniques.

## Dataset

The SMS Spam Collection dataset contains 5,572 labelled SMS messages:

* Ham = Normal message
* Spam = Spam message

## Approach

1. Loaded and explored the SMS dataset
2. Cleaned and preprocessed text data
3. Converted text into TF-IDF features
4. Trained machine learning models
5. Evaluated models using classification metrics
6. Tracked experiments using MLflow
7. Performed 5-Fold Cross Validation
8. Compared Logistic Regression and Naive Bayes
9. Saved the final model and vectorizer
10. Built a prediction script for new messages

## Best Model

TF-IDF + Logistic Regression + Class Balancing (`class_weight="balanced"`)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.971 |
| Precision | 0.878 |
| Recall    | 0.913 |
| F1 Score  | 0.895 |

## Cross Validation

5-Fold Cross Validation Results:

* 0.913
* 0.889
* 0.880
* 0.886
* 0.891

**Mean F1 Score:** 0.892

## Experiments

| Experiment     | Stop Words | Class Weight | F1    | Recall |
| -------------- | ---------- | ------------ | ----- | ------ |
| Baseline       | No         | No           | 0.828 | 0.707  |
| Stop Words     | Yes        | No           | 0.746 | 0.607  |
| Class Balanced | No         | Yes          | 0.891 | 0.873  |
| Threshold 0.3  | Yes        | Yes          | 0.853 | 0.947  |
| Threshold 0.4  | Yes        | Yes          | 0.895 | 0.913  |
| Threshold 0.5  | Yes        | Yes          | 0.896 | 0.860  |
| TF-IDF Bigrams | Yes        | Yes          | 0.874 | 0.947  |
| Naive Bayes    | Yes        | No           | 0.882 | 0.800  |

## Model Comparison

| Model                   | Mean CV F1 |
| ----------------------- | ---------- |
| Logistic Regression     | 0.892      |
| Multinomial Naive Bayes | 0.777      |

Logistic Regression achieved the highest overall performance and was selected as the final model.

## Tech Stack

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Multinomial Naive Bayes
* MLflow
* Joblib

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run predictions:

```bash
python src/predict.py
```

Launch MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
