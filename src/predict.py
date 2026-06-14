import joblib
from preprocess import clean_text

model = joblib.load("../models/spam_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

while True:
    message = input("\nEnter a message or type 'exit': ")

    if message.lower() == "exit":
        break

    cleaned_message = clean_text(message)
    X = vectorizer.transform([cleaned_message])

    prediction = model.predict(X)[0]
    spam_probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        print(f"Prediction: SPAM")
    else:
        print(f"Prediction: HAM")

    print(f"Spam probability: {spam_probability:.2%}")