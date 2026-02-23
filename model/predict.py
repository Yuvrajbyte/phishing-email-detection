import joblib

model = joblib.load("model/svm_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
default_threshold = joblib.load("model/threshold.pkl")

def predict_email(email, threshold):
    vector = vectorizer.transform([email])

    prob = model.predict_proba(vector)[0][1]  # spam prob

    prediction = 1 if prob >= threshold else 0

    return prediction, prob


if __name__ == "__main__":
    print("Spam Detection System")
    print("---------------------")

    user_input = input("Enter an email message: ")
    custom_threshold = input("Enter custom threshold (press Enter to use default): ")

    threshold = (
        float(custom_threshold)
        if custom_threshold.strip() != ""
        else default_threshold
    
    )
    print(default_threshold)



    result, probs = predict_email(user_input, threshold)
   
    print("\n--- Prediction Result ---")
    print("\nSpam Probability:", round(probs * 100, 2), "%")
    print("Ham Probability:", round((1 - probs) * 100, 2), "%")
    print("Prediction:", "Spam" if result == 1 else "Ham")
    
    if result == 1:
        print("\n This email is SPAM!")
    else:
        print("\n This email is NOT spam.")