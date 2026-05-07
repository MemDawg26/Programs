# "Spam Filter" Using Multinomial Naive Bayes
Author: Ethan Lipke  
Date: May 6, 2026  

## Context
This is a project I did for my Artificial Intelligence class along side my friend, Tyner Parker.
We both implemented an Email Spam "Filter" (more along the lines of a detector) using different
models. Obviously, given by the title, I used Multinomial Naive Bayes, while Tyner used Logical
Regression. Before I go any further, at this point, I am not the world's greatest programmer,
especially in regard to AI. However, this project has been fun to work on. I certainly have
learned a lot since starting it, and I honestly feel much more confident in developing this type
of model.

## Why Multinomial Naive Bayes - Example
From my research on the internet and YouTube, there are three main types of Naive Bayes.
Multinomial just happens to be specialized for count-based data (e.g. word count). This means the
model essentially keeps track of the number of times a singular word exists within a particular
classification. Then it creates a probability P(word|label). In the context of spam filtering,
our label is binary -> 1 or 0. 1 is for Spam (and phishing) and 0 is for normal (or 'Ham'). So,
if we have P(money|1), this is essentially read as, "Given that we have a spam email, the
likelihood that the word is "money". Let's consider some training data. If we have 10 spam 
emails, and "money" shows up 7 times in those emails, the P(money|1) = 7/10, or 0.70.

To add on, lets consider another example. We will still have our 10 emails with P(money|1) = 0.70,
but now we have 15 ham emails. In those emails, lets say that "money" shows up 5 times. Then,
our probability P(money|0) = 5/15 = 0.33. Now, let's look at another word - "link". Let's say 
"link" shows up 8 times in the spam emails and 3 times in the ham emails. P(link|1) = 8/10 = 0.80,
and P(link|0) = 3/15 = 0.20. Ok, we now have some data to work with...

So we know that we have 25 emails total. Thus we have probabilities P(0) = 15/25 = 0.6 and P(1) = 
10/25 = 0.40. While this is not the best real-life example, let's say you receive an email that
only contains "money link". You now need to determine whether this email is spam or ham. To do so,
lets first calculate the overall probabilities for the different classifications:

P(1|email) = P(1) * P(money|1) * P(link|1) = 0.4 * 0.7 * 0.8 = 0.224  
P(0|email) = P(0) * P(money|0) * P(link|0) = 0.6 * 0.33 * 0.2 = 0.040

As we can See, the probability that given this email, we have a spam is higher than the probability
that we have a ham. This is how Multinomial Naive Bayes uses word count and probabilities from the
training data to determine the classification of an email.

## Building the Model (train.py)
### Gathering Data
The first thing we need before we start building the model is data. It does not have to be "great"
data, but it does need to be good. Tyner found a pretty solid dataset on Kaggle.com published by
Naser Abdullah Alam. There are 7 files you could work with in this data set:

CEAS_08.csv -> sender, receiver, date, subject, body, urls  
Enron.csv -> subject, body  
Ling.csv -> subject, body  
Nazario.csv -> sender, receiver, date, subject, body, urls  
Nigerian_Fraud.csv -> sender, receiver, date, subject, body, urls  
SpamAssasin.csv -> sender, receiver, date, subject, body, urls  
phishing_email.csv -> combination of previous six files

Either use the first six and exclude the last one, or use the last file on its own. I made the
mistake of using all seven files the first time I programmed this, and it severely affected the 
accuracy of my model (75%).

### Code
Now, my code for this train.py is just about the exact same as Tyner's code for Logical Regression.
The only difference is that we call a different model function, so I felt like I ought to do things a
little bit differently. Thats what train2.py is.

The main differences between train and train2 is that train2:  
- uses six files instead of just the phishing_email.csv
- does not use the custom clean function and instead replaces all NaN's with "" (empty strings)
#### *Important note about MultinomialNB from scikit: It does not accept NaN, so they have to go!
So, in train.py, you see that this is not something we really have to worry about, but we do clean the
data with a clean function. In train2.py, I took a different approach and instead, for each attribute,
I filled all NaN positions with empty strings.

One thing I also found is that having a URL does not have much influence on the decision-making of the
model, so I exclude it from the training data in train2.py. You should see that I only considered the
sender, the subject, and the body. Date/time is also irrelevant, as you are not necessarily more likely
to receive spam on any specific day. Though you could argue months like December or June may see more
ads from department stores and sports + outdoors. Nonetheless, my model worked out fine enough without
it.

If you are using the phishing_email.csv file like in train.py, you don't really have to mess around too
much with the data. You just create your vectors and plug them into the functions, quick and easy. You
may notice some extra random code in my train.py. This is code I wrote to get very specific output for my
AI class project. You can disregard, but it essentially shows the probabilities for the top 10 words
associated with spam emails from this dataset. In short, the probability, logarithmic probability, and
influence score are just a few different ways to visualize how particular words play a role in the model's 
decision making.

Other than that, there is not too much to say about the model. Hopefully, my comments within the code are
detailed enough to somewhat provide an explanation for each step.

## Training and Testing Results
### Classification Report for train.py:  

Accuracy: 0.9786034670869196  

                  precision    recall  f1-score   support

               0       0.97      0.99      0.98      7935
               1       0.99      0.97      0.98      8563

        accuracy                           0.98     16498
       macro avg       0.98      0.98      0.98     16498
    weighted avg       0.98      0.98      0.98     16498

### Classification Report for train2.py:  

Accuracy: 0.9794520547945206  

                  precision    recall  f1-score   support

               0       0.97      0.99      0.98      7897
               1       0.99      0.97      0.98      8601

        accuracy                           0.98     16498
       macro avg       0.98      0.98      0.98     16498
    weighted avg       0.98      0.98      0.98     16498

As we can see, there was very slight improvement in accuracy on train2.py. While it may not make much of
a difference, I opted to use this model for scam_app.py, which we will now get into.

## Building spam_app.py
You may have noticed in both train.py and train2.py that there is a section of code at the bottom that
appears to be writing to some files. Specifically, these are joblib files that store the states of the
trained model and vectorization. Imagine that after you trained your model, you wanted to deploy the model
or take user input to check your own emails for spam/phishing. Well, if you had to train the model every
time you wanted to check an email, it would get tiresome having to wait a minute or two before you can even
input anything.

I wanted to be able to check my own emails for spam and not have to train the model over and over. So, the
code at the bottom of train.py/train2.py stores the model and vectorization states. Then, I created a new 
file (spam_app.py) to load in the already trained model. Now, it takes maybe 5-10 seconds to load in the 
model and start taking input. My spam_app.py takes in a sender's email, a subject line, and a body, sticks
it together and analyzes based on the training data. Note that when you put the data together to make a
prediction, it must be in the same format. So if in the training you had sender, subject, body, and url,
your vector in the new file must also collect the same information.

## Future Project
My next plan is to work on a web-based GUI for the spam_app. While I don't mind running in terminal, I think
having a GUI would just be a cool thing to show off. But more importantly, the whole point of Spam/Phishing
detection is for filtering, which should be an automated process. So, if my skills get a little more
sophisticated, I would like to try to deploy the model through an API, but that may not be something for a
a while.
