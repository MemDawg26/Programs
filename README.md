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
only contains "money link". You now need to determine wether this email is spam or ham. To do so,
lets first calculate the overall probabilities for the different classifications:

P(1|email) = P(1) * P(money|1) * P(link|1) = 0.4 * 0.7 * 0.8 = 0.224  
P(0|email) = P(0) * P(money|0) * P(link|0) = 0.6 * 0.33 * 0.2 = 0.040

As we can See, the probability that given this email, we have a spam is higher than the probability
that we have a ham. This is how Multinomial Naive Bayes uses word count and probabilities from the
training data to determine the classification of an email.

## Building the Model (train.py)
### - Gathering Data
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

Either use the first six and exclude the last one, or use the last file on its own. I made that
mistake the first time I programmed this, and it severely affected the accuracy of my model (75%).
