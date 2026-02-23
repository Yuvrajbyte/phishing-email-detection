import joblib
import pandas as pd
from preprocessing.clean_text import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

df = df[['v1','v2']]
df.columns = ['label', 'message']

df['cleaned_message'] = df['message'].apply(clean_text) #for cleaning

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=2)          #Vectorization
X = vectorizer.fit_transform(df['cleaned_message'])

y = df['label']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC


nb_model = MultinomialNB()                                                                              # Train Naive Bayes
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=2000, class_weight='balanced', C=2)
lr_model.fit(X_train, y_train)


nb_preds = nb_model.predict(X_test)                                                                     #Predictions
lr_preds = lr_model.predict(X_test)

svm_model = SVC(kernel='linear', probability=True, class_weight='balanced')
svm_model.fit(X_train, y_train)

import numpy as np
from sklearn.metrics import f1_score

y_probs = svm_model.predict_proba(X_test)[:, 1]

best_threshold = 0
best_f1 = 0

for threshold in np.linspace(0.0, 1.0, 101):
    y_pred = (y_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("Optimal Threshold:", round(best_threshold, 2))
print("Best F1 Score:", round(best_f1, 4))

y_pred_svm = (y_probs >= best_threshold).astype(int)

print("\n===== SVM Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

print("===== Naive Bayes Results =====")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print(classification_report(y_test, nb_preds))

print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print(classification_report(y_test, lr_preds))


import joblib
joblib.dump(best_threshold, "model/threshold.pkl")
joblib.dump(svm_model, "model/svm_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Best model and vectorizer saved successfully!")