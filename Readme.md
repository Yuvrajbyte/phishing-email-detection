Phishing Email Detection Using Machine Learning

🚀Overview
The projects implements a spam/phishing email detection system using various ML techniques.
It compares multiple classifiers and optimizes threshold to maximize F1-score under class imbalance

The final deployed model uses an optimized Support Vector Machine(SVM) with probability based classification, other techniques were linear regression and Naive Bayes.

Key Features 
->Text processing
->TF-IDF vectorization
->Model comparision:
    Support Vector Machine
    Naive Bayes
    Logic Regression
->Class Imbalance Handling(class_weight = 'balanced')
->Automatic thresold optimization(To maximize F1)
->Probability based inference
