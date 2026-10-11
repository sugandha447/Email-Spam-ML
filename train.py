import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("dataset/email_data.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# Convert ham/spam into numbers
encoder = LabelEncoder()
df["Category"] = encoder.fit_transform(df["Category"])

# Input and output
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

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

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

print("\nAll four models trained successfully!")