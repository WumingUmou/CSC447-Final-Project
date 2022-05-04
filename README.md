# CSC447 Final Project
 CSC447 Final Project of Deyao Kong

# Introduction
This project is a implementation of Google's BERT model onto a Kaggle compitation dataset.

## Compitation
The goal of this compitation is using some general information of a twitter to predict if the sender's political party is Democratic or Republican.

The dataset include:

| Column        | Description          | 
| :------------- |:------------- |
| favorite_count  | the count of favourite for this twitter | 
| full_text  | the oriangle text for this twitter      | 
| hashtags | the hashtags for this twitter  | 
| retweet_count | how many retweets did this twitter get |
|year| the year this twitter got published |
|party_id| the political party of this twitter's sender |

There are 58k+ rows of data in the full training dataset. For the ease of uploading to github, I made a random sample of 10k and 50k data points, saved as `demo_10000.csv` and `demo_50000.csv` respectivly.


