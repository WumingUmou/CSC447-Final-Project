# CSC447 Final Project
 CSC447 Final Project of Deyao Kong

# Introduction
This project is a implementation of Google's BERT model onto a Kaggle compitation dataset.

Link to the Kaggle compitation:
https://www.kaggle.com/competitions/congressionaltweetcompetitionspring2022/overview
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
|party_id **(target)**| the political party of this twitter's sender |

There are 58k+ rows of data in the full training dataset. For the ease of uploading to github, I made a random sample of 10k and 50k data points, saved as `demo_10000.csv` and `demo_50000.csv` respectivly.

I also made a random sample of 2k data points as prediction set `test.csv`

# Using the model

`python3 bert_Twitter.py [input file] [output file] [N_EPOCH] [BATCH_SIZE] [MAX_TOKEN_LEN]`

||||
| ------ | ------ | ------ |
|`input file`|`string`| for demo, choose from `demo_10000.csv` or `demo_50000.csv`|
|`output file`|`string`| for demo, choose `test.csv`|
|`N_EPOCH`| `int`| the total number of epoches that the model will run|
|`BATCH_SIZE`|`int`| the size of each batch, choose by RAM usage|
|`MAX_TOKEN_LEN`| `int` | the max length of a tokenized vector|

The result will be saved to `output file`

## Environment Requirements
    GPU with CUDA installed
    CUDA = 11.3
    python = 3.9
    pytorch = 1.11.0 with gpu
    pandas = 1.4.2
    numpy = 1.21.5
    tqdm = 4.64.0
    transformers = 4.18.0
    scikit-learn = 1.0.2
    seaborn = 0.11.2
    matplotlib = 3.5.1
    pytorch_lightning = 1.6.2
    re = 2.2.1

# Results
The model gave a `0.89` percision in the full test set, trained on the full train set after 8 epoches.

However, on the `demo_10000.csv`, the BCELoss raised to `0.9` on the validation set and `0.03` on the training set after the 5th epoch due to extreame overfitting problem where the size of data for training is too small for the model.

# reference
https://github.com/codertimo/BERT-pytorch