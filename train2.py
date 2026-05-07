#import packages
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

#import data file
DATA_PATH = "data/"

files = ["CEAS_08.csv",
         "Enron.csv",
         "Ling.csv",
         "Nazario.csv",
         "Nigerian_Fraud.csv",
         "SpamAssasin.csv"
]

# read in files
dfs = []
for file in files:
  print("Reading in file: " + file + "...")
  df = pd.read_csv(DATA_PATH + file)
  dfs.append(df)

print("All files read successfully\n")

df = pd.concat(dfs, ignore_index=True)

#clean data
df["text"] = df["sender"].fillna("") + " " + df["subject"].fillna("") + " " + df["body"].fillna("")

#vectorize the attributes 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
Y = df["label"]

#split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
model = MultinomialNB(alpha=0.3)
model.fit(X_train, y_train)

#evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='g', cmap='BuGn')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

#save and dump trained model
from joblib import dump

dump(vectorizer, "scam_NB_vec.joblib")
#dump(subject_vec, "subject_vec.joblib")
#dump(body_vec, "body_vec.joblib")
dump(model, "scam_NB_model.joblib")