import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


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


# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)


# Take new email from user
message = input("Enter your email message: ")


# Convert new message into numbers
message_vector = vectorizer.transform([message])


# Predict
prediction = model.predict(message_vector)


# Display result
if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")