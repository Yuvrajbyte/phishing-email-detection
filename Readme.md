Phishing Email Detection Using Machine Learning

OVERVIEW
The projects implements a spam/phishing email detection system using various ML techniques.
It compares multiple classifiers and optimizes threshold to maximize F1-score under class imbalance

The final deployed model uses an optimized Support Vector Machine(SVM) with probability based classification, other techniques were linear regression and Naive Bayes.

KEY FEATURES 
    Text processing
    TF-IDF vectorization
    Model comparision:
        Support Vector Machine
        Naive Bayes
        Logic Regression
    Class Imbalance Handling(class_weight = 'balanced')
    Automatic thresold optimization(To maximize F1)
    Probability based inference
    Persistent trained artifacts(model, vectorizer, threshold)

MODEL PERFORMANCE
                                    Accuracy    Spam f1 Score
        SVM(optimized)              98%         0.92    
        Naive Bayes                 97%         0.89
        Linear Regression           97%         0.90

Optimal decision threshold: 0.49

Why Threshold Optimization?
Instead of using the default 0.5 probability threshold, this project computes an optimal threshold that maximizes F1 score

HOW TO RUN
1️. Install dependencies
    pip install -r requirements.txt
2️. Train the model
    python train_model.py
3️. Run prediction
    python predict.py

You can optionally override the threshold during prediction.

Example Output
    Spam Probability: 52.14 %
    Ham Probability: 47.86 %
    Threshold Used: 0.49
    Prediction: Spam

TECH STACK
    Python
    scikit-learn
    NumPy
    Pandas
    Joblib

FUTURE IMPROVEMENTS
    ROC curve visualization
    Flask API deployment
    Streamlit web interface
    Cross-validation tuning
    Hyperparameter optimization

What This Project Demonstrates
    End-to-end ML pipeline
    Model evaluation and comparison
    Handling class imbalance
    Decision boundary tuning
    Production-consistent inference design