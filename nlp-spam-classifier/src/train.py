import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

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
threshold = 0.3

#stop words ngrams
#vectorizer = TfidfVectorizer(stop_words=stop_words)
ngram_range = (1,2)
vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=ngram_range
)
X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
run_name = "naivebayes"
with mlflow.start_run(run_name=run_name):
    print("MLflow run started:", mlflow.active_run().info.run_id)
    # model = LogisticRegression(
    #     class_weight=class_weight,
    #     max_iter=1000
    # )
    model = MultinomialNB()
    scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
    )

    print("Fold Scores:")
    print(scores)

    print("\nMean F1:")
    print(scores.mean())

    model.fit(X_train, y_train)

    # feature_names = vectorizer.get_feature_names_out()

    # weights_df = pd.DataFrame({
    #     "word": feature_names,
    #     "weight": model.coef_[0]
    # })

    # print("\nTop 20 SPAM words:")
    # print(
    #     weights_df
    #     .sort_values("weight", ascending=False)
    #     .head(20)
    # )

    # print("\nTop 20 HAM words:")
    # print(
    #     weights_df
    #     .sort_values("weight")
    #     .head(20)
    # )

    #predictions = model.predict(X_test)
    spam_probs = model.predict_proba(X_test)[:, 1]

    predictions = (spam_probs >= threshold).astype(int)
    metrics = evaluate_model(
        y_test,
        predictions
    )
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("stop_words", stop_words)
    mlflow.log_param("class_weight", class_weight)
    mlflow.log_param("model", "naive_bayes")
    mlflow.log_param("ngram_range", str(ngram_range))
    mlflow.log_param("threshold", threshold)

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



joblib.dump(model, "../models/spam_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")