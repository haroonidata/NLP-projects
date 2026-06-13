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

stop_words = None
class_weight = None

vectorizer = TfidfVectorizer(stop_words=stop_words)

X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():
    print("MLflow run started:", mlflow.active_run().info.run_id)
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("stop_words", stop_words)
    mlflow.log_param("class_weight", class_weight)
    mlflow.log_param("model", "logistic_regression")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("spam_precision", precision)
    mlflow.log_metric("spam_recall", recall)
    mlflow.log_metric("spam_f1", f1)

    mlflow.sklearn.log_model(model, "spam_model")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Spam precision: {precision:.4f}")
    print(f"Spam recall: {recall:.4f}")
    print(f"Spam F1: {f1:.4f}")

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print("MLflow run finished")