import re

import nltk
import pandas as pd
from pandas import DataFrame

def pre_process(data: DataFrame) -> DataFrame:
    for row in data:
        msg = row.lower()
        msg = replace_symbols(msg)
        msg = remove_punctuation(msg)
        msg = remove_stop_words(msg)
        msg = remove_whitespace(msg)
        msg = stem(msg)
        data = data.replace(row, msg)
    return data

def replace_symbols(msg: str) -> str:
    # Normalises the input data by generifying phone numbers, email address, etc.
    regexes = [
        # Removes spaces between numbers for phone number processing
        (r'(?<=\d) +(?=\d)', ''),
        # Replaces email addresses
        (r'[\w\-.]+@([\w\-]+\.)+[\w\-]{2,10}', 'emailAddress'),
        # Replaces domains and urls (matching all possible characters in urls)
        (r'\b(https?:\/\/)?[\/\w\-_\$\.\+!\*\'\(\)\,]+\.(\w{2,10})[\/\w\-_\$\.\+!\*\'\(\)\,]*', 'urlAddress'),
        # Replaces numbers/decimals
        (r'\b\d+(\.\d+)?\b', 'numberSymbol'),
        # Replaces currency symbols
        (r'[£$€]', 'currencySymbol'),
        # Replaces phone numbers (inc. country code and hyphens)
        (r'(\+\d{1,3})?(\s\()?\d{3}(\)\s)?\d{7,8}', 'phoneNumber')
    ]

    for regex, replacement in regexes:
        msg = re.sub(regex, replacement, msg)

    return msg


def remove_whitespace(msg: str) -> str:
    processed = re.sub(r'\s+', ' ', msg)
    processed = re.sub(r'(^\s+|\s+$)', '', processed)
    return processed

def remove_punctuation(msg: str) -> str:
    return re.sub(r'[^\w\d\s]', '', msg)

def remove_stop_words(msg: str) -> str:
    # Removes stop words which do not contribute meaning in English phrases
    stop_words = nltk.corpus.stopwords.words('english')
    processed = []

    for word in msg.split(' '):
        if word not in set(stop_words):
            processed.append(word)

    return ' '.join(processed)

def stem(msg: str) -> str:
    # Removes suffixes to reduce words to only their stem
    stemmer = nltk.PorterStemmer()
    return ' '.join(stemmer.stem(term) for term in msg.split())

def main():
    data_set = pd.read_table('data/sms-dataset', header=None)
    pre_processed = pre_process(data_set[1])

main()