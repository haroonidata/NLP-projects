import pandas as pd
import mlflow
import mlflow.sklearn

from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from evaluate import evaluate_model
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("nlp-spam-classifier")

df = pd.read_csv("../data/spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df["clean_message"] = df["message"].apply(clean_text)

stop_words = "english"
class_weight = "balanced"

vectorizer = TfidfVectorizer(stop_words=stop_words)

X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run(run_name="threshold_0.3"):
    print("MLflow run started:", mlflow.active_run().info.run_id)
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    #predictions = model.predict(X_test)
    spam_probs = model.predict_proba(X_test)[:, 1]

    predictions = (spam_probs >= 0.3).astype(int)

    metrics = evaluate_model(
        y_test,
        predictions
    )
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("stop_words", stop_words)
    mlflow.log_param("class_weight", class_weight)
    mlflow.log_param("model", "logistic_regression")

    mlflow.log_metric(
    "accuracy",
    metrics["accuracy"]
    )

    mlflow.log_metric(
        "spam_precision",
        metrics["precision"]
    )

    mlflow.log_metric(
        "spam_recall",
        metrics["recall"]
    )

    mlflow.log_metric(
        "spam_f1",
        metrics["f1"]
    )
    mlflow.sklearn.log_model(model, "spam_model")

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print("MLflow run finished")

import joblib

joblib.dump(model, "../models/spam_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")