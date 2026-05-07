import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from rich.prompt import Prompt
from rich import print
from joblib import load

print("Loading Files...")
model = load("scam_NB_model.joblib")
vec = load("scam_NB_vec.joblib")

#function to enter email attributes
def check_email():
    sender = Prompt.ask("[yellow]Please enter the sender's address")
    subject = Prompt.ask("[yellow]Enter the subject line")
    body = Prompt.ask("[yellow]Enter the body")

    message = sender + subject + body
    x_message = vec.transform([message])
    prediction = model.predict(x_message)
    print()

    if prediction[0] == 1:
        print("[bold red]Warning! [/bold red][yellow]This email has been detected as a spam or phishing email. Be Cautious!\n")
    else:
        print("[bold green]This email has been detected as a normal email, though it is good to still remain cautious.\n")

while(1):
    ans = Prompt.ask("[yellow]Would you like to test an email?(y/n)")
    if ans == "y":
        #some function
        check_email()
    elif ans == "n":
        break
    else:
        print("Invalid answer. Try Again.\n")
