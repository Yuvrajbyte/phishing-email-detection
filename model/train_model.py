import joblib
import pandas as pd
from preprocessing.clean_text import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:\Python Projects\.vscode\Phishing Email Detection\dataset\spam.csv', encoding="latin-1")

df = df[['v1','v2']]
df.columns = ['label', 'message']

df['cleaned_message'] = df['message'].apply(clean_text) #for cleaning

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=2)          #Vectorization
X = vectorizer.fit_transform(df['cleaned_message'])

y = df['label']

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

y_pred_svm = svm_model.predict(X_test)

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

joblib.dump(svm_model, "model/svm_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Best model and vectorizer saved successfully!")