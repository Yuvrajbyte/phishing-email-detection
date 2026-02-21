import joblib

model = joblib.load("model/svm_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_email(email):
    vector = vectorizer.transform([email])

    prob = model.predict_proba(vector)[0][1]  # spam prob

    prediction = 1 if prob > 0.4 else 0

    return prediction, prob


if __name__ == "__main__":
    print("Spam Detection System")
    print("---------------------")

    user_input = input("Enter an email message: ")

    result, probs = predict_email(user_input)
   
    print("\n--- Prediction Result ---")
    print("\nSpam Probability:", round(probs * 100, 2), "%")
    print("Ham Probability:", round((1 - probs) * 100, 2), "%")
    print("Prediction:", "Spam" if result == 1 else "Ham")
    
    if result == 1:
        print("\n This email is SPAM!")
    else:
        print("\n This email is NOT spam.")