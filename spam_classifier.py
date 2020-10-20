import re
import nltk
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

def pre_process_msg(msg: str) -> str:
    msg = msg.lower()
    msg = replace_symbols(msg)
    msg = remove_punctuation(msg)
    msg = remove_stop_words(msg)
    msg = remove_whitespace(msg)
    return stem(msg)

def pre_process(data: DataFrame) -> DataFrame:
    for row in data:
        data = data.replace(row, pre_process_msg(row))
    return data

# Normalises the input data by generifying phone numbers, email address, etc.
def replace_symbols(msg: str) -> str:
    regexes = [
        # Removes spaces between numbers for phone number processing
        (r'(?<=\d) +(?=\d)', ''),
        # Replaces email addresses
        (r'[\w\-.]+@([\w\-]+\.)+[\w\-]{2,10}', 'emailAddress'),
        # Replaces domains and urls (matching all possible characters in urls)
        (r'\b(https?:\/\/)?[\/\w\-_\$\.\+!\*\'\(\)\,]+\.(\w{2,10})[\/\w\-_\$\.\+!\*\'\(\)\,]*', 'urlAddress'),
        # Replaces phone numbers (inc. country code and hyphens)
        (r'(\+\d{1,3})?(\s\()?\d{3}(\)\s)?\d{7,8}', 'phoneNumber'),
        # Replaces numbers/decimals
        (r'\b\d+(\.\d+)?\b', 'numberSymbol'),
        # Replaces currency symbols
        (r'[£$€]', 'currencySymbol')
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

# Removes stop words which do not contribute meaning in English phrases
def remove_stop_words(msg: str) -> str:
    stop_words = nltk.corpus.stopwords.words('english')
    processed = []

    for word in msg.split(' '):
        if word not in set(stop_words):
            processed.append(word)

    return ' '.join(processed)

# Removes suffixes to reduce words to only their stem
def stem(msg: str) -> str:
    stemmer = nltk.PorterStemmer()
    return ' '.join(stemmer.stem(term) for term in msg.split())

def classify_msgs(classifier: LinearSVC, vectorizer: TfidfVectorizer):
    msg = input("Please enter a message to classify. Input \'q\' to exit the program. ")

    while msg != "q":
        is_spam = classifier.predict(vectorizer.transform([pre_process_msg(msg)]))

        if is_spam:
            print("This message is spam.")
        else:
            print("This message is not spam.")

        msg = input("Please enter another message to classify: ")

# Estimates the accuracy of this model and prints this data to stdout
def calculate_statistics(classifier: LinearSVC, msg_test: csr_matrix, label_test: csr_matrix):
    label_pred = classifier.predict(msg_test)
    print("Model accuracy: {}".format(metrics.f1_score(label_test, label_pred)))

    print("Confusion matrix: \n{}".format(pd.DataFrame(
        metrics.confusion_matrix(label_test, label_pred),
        index=[['actual', 'actual'], ['spam', 'ham']],
        columns=[['predicted', 'predicted'], ['spam', 'ham']]
    )))

def main():
    data_set = pd.read_table('data/sms-dataset', header=None)
    label_encoder = LabelEncoder()
    # Encodes labels as 'spam' = 1 and 'ham' = 0
    encoded_labels = label_encoder.fit_transform(data_set[0])

    # Normalises the data and performs stemming
    pre_processed = pre_process(data_set[1])

    # Performs bigram tokenization on messages and computes the tf-idf statistic
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    bigrams = vectorizer.fit_transform(pre_processed)

    # Splits data into training and testing data with a 80/20 split
    msg_train, msg_test, label_train, label_test = train_test_split(bigrams, encoded_labels, test_size=0.2,
                                                                    random_state=42, stratify=encoded_labels)

    # Trains model with SVM
    classifier = LinearSVC(loss="hinge")
    classifier.fit(msg_train, label_train)
    classify_msgs(classifier, vectorizer)

    calculate_statistics(classifier, msg_test, label_test)

main()