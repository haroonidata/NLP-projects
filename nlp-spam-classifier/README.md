# NLP Spam Classifier

## Project Overview
This project builds a machine learning model to classify SMS messages as spam or ham.

## Dataset
The dataset contains SMS messages labelled as:
- ham = normal message
- spam = spam message

## Approach
1. Loaded SMS dataset
2. Cleaned text data
3. Converted text to TF-IDF features
4. Trained Logistic Regression model
5. Evaluated using Accuracy, Precision, Recall and F1
6. Tracked experiments using MLflow
7. Saved best model
8. Built prediction script

## Best Model
The best model was Logistic Regression with class balancing.

| Metric | Score |
|---|---:|
| Accuracy | 0.971 |
| Precision | 0.878 |
| Recall | 0.913 |
| F1 Score | 0.895 |

## Experiments
| Experiment | Stop Words | Class Weight | F1 | Recall |
|---|---|---|---:|---:|
| Baseline | No | No | 0.828 | 0.707 |
| Stop Words | Yes | No | 0.746 | 0.607 |
| Class Balanced | No | Yes | 0.895 | 0.913 |

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt

Train model

python src/train.py

Run prediction:
python src/predict.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
##
Tech Stack
Python
Pandas
Scikit-learn
TF-IDF
Logistic Regression
MLflow
Joblib