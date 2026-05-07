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
df = pd.read_csv("data/phishing_email.csv")

#function that cleans the text
def clean_text(text):
  text = str(text).lower()
  text = re.sub(r"http\S+", "", text)
  text = re.sub(r"\d+", "", text)
  text = re.sub(r"[^\w\s]", "", text)
  return text

df['text_combined'] = df['text_combined'].apply(clean_text)

#vectorize the text_combines column, which contains all of the information 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text_combined'])
y = df["label"]

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = MultinomialNB(alpha=0.3)
model.fit(X_train, y_train)

#evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import numpy as np

feature_names = vectorizer.get_feature_names_out()

ham_probs = model.feature_log_prob_[0]
spam_probs = model.feature_log_prob_[1]

influence = spam_probs - ham_probs

df2 = pd.DataFrame({
    "word": feature_names,
    "ham_log_prob": ham_probs,
    "spam_log_prob": spam_probs,
    "influence_score": influence,
    "ham_prob": np.exp(ham_probs),
    "spam_prob": np.exp(spam_probs)
})

print(df2.sort_values("influence_score", ascending=False).head(10))

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