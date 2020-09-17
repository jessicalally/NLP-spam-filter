import pandas as pd
from pandas import DataFrame
import re
import nltk

def separate_msgs(data_set: DataFrame) -> (DataFrame, DataFrame):
    ham_txt = pd.DataFrame(columns=['Message'])
    spam_txt = pd.DataFrame(columns=['Message'])

    for index, column in data_set.iterrows():
        label = column[0]
        msg = column[1]

        if label == "ham":
            ham_txt = ham_txt.append({'Message': msg}, ignore_index=True)
        elif label == "spam":
            spam_txt = spam_txt.append({'Message': msg}, ignore_index=True)
        else:
            raise ValueError("Invalid label {} on line {}".format(label, index))

    return ham_txt, spam_txt

def normalize(data: DataFrame) -> DataFrame:
    for index, column in data.iterrows():
        msg = column[0]
        msg = replace_symbols(msg)
        msg = remove_punctuation(msg)
        msg = remove_stop_words(msg)
        msg = msg.lower()
        msg = remove_whitespace(msg)
        msg = stemmed(msg)
        data = data.replace(column, msg)
    return data

def replace_symbols(msg: str) -> str:
    processed = re.sub(r'(?<=\d) +(?=\d)', '', msg)
    processed = re.sub(r'[\w\-.]+@([\w\-]+\.)+[\w\-]{2,4}', 'emailaddr', processed)
    processed = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', processed)
    processed = re.sub(r'[£$]', 'moneysymb', processed)
    processed = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', processed)
    processed = re.sub(r'\d+(\.\d+)?', 'numbr', processed)
    return processed

def remove_whitespace(msg: str) -> str:
    processed = re.sub(r'\s+', ' ', msg)
    processed = re.sub(r'^\s+|\s+?$', '', processed)
    return processed

def remove_punctuation(msg: str) -> str:
    return re.sub(r'[^\w\d\s]', '', msg)

def remove_stop_words(msg: str) -> str:
    stop_words = nltk.corpus.stopwords.words('english')
    return ' '.join(term for term in msg.split() if term not in set(stop_words))

def stemmed(msg: str) -> str:
    stemmer = nltk.PorterStemmer()
    return ' '.join(stemmer.stem(term) for term in msg.split())

def main():
    data_set = pd.read_table('data/SMSSpamCollection.txt', header=None)
    (ham_txt, spam_txt) = separate_msgs(data_set)
    ham_txt = normalize(ham_txt)
    spam_txt = normalize(spam_txt)

main()