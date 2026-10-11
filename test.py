import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("dataset/email_data.csv")

# Convert ham/spam into numbers
encoder = LabelEncoder()
df["Category"] = encoder.fit_transform(df["Category"])

X = df["Message"]
y = df["Category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into numbers
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train models
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

model_svm = SVC()
model_svm.fit(X_train, y_train)

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# Predictions
y_pred_lr = model_lr.predict(X_test)
y_pred_svm = model_svm.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_nb = model_nb.predict(X_test)


# Evaluation function
def evaluate_model(name, y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# Evaluate all models
evaluate_model("Logistic Regression", y_test, y_pred_lr)

evaluate_model("SVM", y_test, y_pred_svm)

evaluate_model("Random Forest", y_test, y_pred_rf)

evaluate_model("Naive Bayes", y_test, y_pred_nb)