import ast
import logging
import string
import numpy as np
import codecs

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from pymystem3 import Mystem

CORPUS_PATH = 'data/corpus.txt'
POSITIVE_PATH = 'data/positive.txt'
NEGATIVE_PATH = 'data/negative.txt'

log = logging.getLogger()

POS_MARK = 1
NEG_MARK = 0


def read_data(corpus_path):
    with open(corpus_path, 'r') as f:
        log.info("Reading file...")

        reviews = []
        marks = []

        lines = f.readlines()
        lines = [x.strip() for x in lines]

        for line in lines:
            dictionary = ast.literal_eval(line)
            reviews.append(dictionary['description'])

            recom_author_mark = dictionary['recom_author_mark']
            marks.append(parse_label(recom_author_mark))

    return reviews, marks


def parse_label(recom_author_mark):
    if recom_author_mark == 'ДА':
        return POS_MARK
    elif recom_author_mark == '':
        return NEG_MARK
    else:
        raise Exception("Unknown recom_author_mark")


def split_reviews(reviews, marks, n=500):
    pos_reviews = []
    neg_reviews = []

    log.info(f"Split reviews to {n} positive and {n} negative.")

    for index, review in enumerate(reviews):
        if marks[index] == POS_MARK and len(pos_reviews) < n:
            pos_reviews.append(review)
        elif marks[index] == NEG_MARK and len(neg_reviews) < n:
            neg_reviews.append(review)

        if len(pos_reviews) == n and len(neg_reviews) == n:
            break

    return pos_reviews, neg_reviews


def tokenize(document):
    ignore = set(stopwords.words('russian'))
    stem = Mystem()

    tokens = stem.lemmatize(document)

    tokens = [w.lower() for w in tokens if w not in ignore]
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w.isalpha()]

    return tokens


def tokenize_documents_extend(documents):
    texts = []

    for document in documents:
        w = tokenize(document)
        texts.extend(w)

    return texts


def tokenize_documents_append(documents):
    texts = []

    for document in documents:
        w = tokenize(document)
        texts.append(w)

    return texts


def print_score_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = precision_recall_fscore_support(y_pred=y_pred, y_true=y_test, average='binary', pos_label=POS_MARK)
    log.info(f'Precision: {round(metrics[0], 3)}, recall: {round(metrics[1], 3)}, f-measure: {round(metrics[2], 3)}')


def feature_importances(X, Y):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    return model.feature_importances_


def feature_importances_gridsearch(X, Y):
    model = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid={})
    model.fit(X, Y)
    return model.best_estimator_.feature_importances_


def print_most_valuable_features(tokens, tokens_importances, n=100):
    tokens_importances, tokens = zip(*sorted(zip(tokens_importances, tokens), reverse=True))

    print(tokens[:n])
    print(tokens_importances[:n])


def setup_logger():
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s')
    log.setLevel(logging.INFO)


def read_words(filepath):
    with codecs.open(filepath, 'r', 'utf-8') as f:
        words = f.readlines()
        words = [x.strip() for x in words]
        return set(words)


def sentimental_values(documents, positive_words, negative_words):
    sentimental_values = []

    for document in documents:
        sentimental_value = 0
        for token in document:
            if token in positive_words:
                sentimental_value += 1
            elif token in negative_words:
                sentimental_value -= 1

        sentimental_values.append(sentimental_value)

    return sentimental_values


def fit(x_train, y_train, x_test, y_test):
    model = LogisticRegression(solver="lbfgs")
    model.fit(x_train, y_train)
    print_score_model(model, x_test, y_test)


def get_documents_tokens(x_train, x_test):
    train_tokens = tokenize_documents_append(x_train)
    test_tokens = tokenize_documents_append(x_test)
    return train_tokens, test_tokens


def get_char_ngram_feature(x_train, x_test):
    vectorizer = TfidfVectorizer(max_features=50000, min_df=5, analyzer='char', ngram_range=(3, 3))
    X_train_vect_char_ngram = vectorizer.fit_transform(x_train)
    X_test_vect_char_ngram = vectorizer.transform(x_test)

    return X_train_vect_char_ngram, X_test_vect_char_ngram


def get_word_ngram_feature(x_train, x_test):
    vectorizer = TfidfVectorizer(max_features=50000, min_df=5, analyzer='word', ngram_range=(3, 3))
    X_train_vect_word_ngram = vectorizer.fit_transform(x_train)
    X_test_vect_word_ngram = vectorizer.transform(x_test)

    return X_train_vect_word_ngram, X_test_vect_word_ngram


def get_word_count_feature(train_tokens, test_tokens):
    train_word_counts = [len(tokens) for tokens in train_tokens]
    train_word_counts = np.reshape(train_word_counts, (-1, 1))

    test_word_counts = [len(tokens) for tokens in test_tokens]
    test_word_counts = np.reshape(test_word_counts, (-1, 1))

    return train_word_counts, test_word_counts


def get_char_count_feature(X_train, X_test):
    train_char_counts = [len(document) for document in X_train]
    train_char_counts = np.reshape(train_char_counts, (-1, 1))

    test_char_counts = [len(document) for document in X_test]
    test_char_counts = np.reshape(test_char_counts, (-1, 1))

    return train_char_counts, test_char_counts


def get_bracket_count_feature(X_train, X_test):
    train_bracket_counts = [document.count(')') - document.count('(') for document in X_train]
    train_bracket_counts = np.reshape(train_bracket_counts, (-1, 1))

    test_bracket_counts = [document.count(')') - document.count('(') for document in X_test]
    test_bracket_counts = np.reshape(test_bracket_counts, (-1, 1))

    return train_bracket_counts, test_bracket_counts


def get_sentimental_feature(train_tokens, test_tokens):
    positive_words = read_words(POSITIVE_PATH)
    negative_words = read_words(NEGATIVE_PATH)

    train_sentimental_values = sentimental_values(train_tokens, positive_words, negative_words)
    train_sentimental_values = np.reshape(train_sentimental_values, (-1, 1))

    test_sentimental_values = sentimental_values(test_tokens, positive_words, negative_words)
    test_sentimental_values = np.reshape(test_sentimental_values, (-1, 1))

    return train_sentimental_values, test_sentimental_values


if __name__ == '__main__':
    setup_logger()

    reviews, marks = read_data(CORPUS_PATH)
    pos_reviews, neg_reviews = split_reviews(reviews, marks)

    y_pos = [1] * len(pos_reviews)
    y_neg = [0] * len(neg_reviews)

    X_train, X_test, y_train, y_test = train_test_split(pos_reviews + neg_reviews, y_pos + y_neg, test_size=0.2)

    vectorizer = TfidfVectorizer(max_features=50000, min_df=5, tokenizer=tokenize)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    fit(X_train_vect.toarray(), y_train, X_test_vect.toarray(), y_test)

    log.info('Features importance:')
    token_importances = feature_importances(X_train_vect.toarray(), y_train)
    word_arr = tokenize_documents_extend(X_train)
    print_most_valuable_features(word_arr, token_importances)

    # Test Features char ngram
    log.info('Features char ngram')
    X_train_vect_char_ngram, X_test_vect_char_ngram = get_char_ngram_feature(X_train, X_test)

    train_matrix = np.append(X_train_vect.toarray(), X_train_vect_char_ngram.toarray(), axis=1)
    test_matrix = np.append(X_test_vect.toarray(), X_test_vect_char_ngram.toarray(), axis=1)

    fit(train_matrix, y_train, test_matrix, y_test)

    # Test Features word ngram
    log.info('Features word ngram')
    X_train_vect_word_ngram, X_test_vect_word_ngram = get_word_ngram_feature(X_train, X_test)

    train_matrix = np.append(X_train_vect.toarray(), X_train_vect_word_ngram.toarray(), axis=1)
    test_matrix = np.append(X_test_vect.toarray(), X_test_vect_word_ngram.toarray(), axis=1)

    fit(train_matrix, y_train, test_matrix, y_test)

    # Test Feature word count
    log.info('Feature word count')
    train_tokens, test_tokens = get_documents_tokens(X_train, X_test)

    train_word_counts, test_word_counts = get_word_count_feature(train_tokens, test_tokens)

    train_matrix = np.append(X_train_vect.toarray(), train_word_counts, axis=1)
    test_matrix = np.append(X_test_vect.toarray(), test_word_counts, axis=1)

    fit(train_matrix, y_train, test_matrix, y_test)

    # Test Feature characters count
    log.info('Feature characters count')
    train_char_counts, test_char_counts = get_char_count_feature(X_train, X_test)

    train_matrix = np.append(X_train_vect.toarray(), train_char_counts, axis=1)
    test_matrix = np.append(X_test_vect.toarray(), test_char_counts, axis=1)

    fit(train_matrix, y_train, test_matrix, y_test)

    # Test Feature brackets count
    log.info('Feature brackets count')
    train_bracket_counts, test_bracket_counts = get_bracket_count_feature(X_train, X_test)

    train_matrix = np.append(X_train_vect.toarray(), train_bracket_counts, axis=1)
    test_matrix = np.append(X_test_vect.toarray(), test_bracket_counts, axis=1)

    fit(train_matrix, y_train, test_matrix, y_test)

    # Test Feature Sentimental
    log.info('Feature Sentimental')
    train_sentimental_values, test_sentimental_values = get_sentimental_feature(train_tokens, test_tokens)

    train_matrix = np.append(X_train_vect.toarray(), train_sentimental_values, axis=1)
    test_matrix = np.append(X_test_vect.toarray(), test_sentimental_values, axis=1)

    fit(train_matrix, y_train, test_matrix, y_test)

    # All futures
    log.info('All futures')
    train_matrix = np.append(X_train_vect.toarray(), X_train_vect_char_ngram.toarray(), axis=1)
    test_matrix = np.append(X_test_vect.toarray(), X_test_vect_char_ngram.toarray(), axis=1)

    train_matrix = np.append(train_matrix, X_train_vect_word_ngram.toarray(), axis=1)
    test_matrix = np.append(test_matrix, X_test_vect_word_ngram.toarray(), axis=1)

    train_matrix = np.append(train_matrix, train_word_counts, axis=1)
    test_matrix = np.append(test_matrix, test_word_counts, axis=1)

    train_matrix = np.append(train_matrix, train_char_counts, axis=1)
    test_matrix = np.append(test_matrix, test_char_counts, axis=1)

    train_matrix = np.append(train_matrix, train_bracket_counts, axis=1)
    test_matrix = np.append(test_matrix, test_bracket_counts, axis=1)

    fit(train_matrix, y_train, test_matrix, y_test)

    # GridSearchCV
    log.info('GridSearchCV features importance:')
    token_importances = feature_importances_gridsearch(train_matrix, y_train)
    print_most_valuable_features(word_arr, token_importances)
