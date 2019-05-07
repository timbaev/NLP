import ast
import logging
import string
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from pymystem3 import Mystem
from gensim.models import Word2Vec

CORPUS_PATH = 'data/corpus.txt'
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


def print_most_valuable_features(tokens, tokens_importances, n=100):
    tokens_importances, tokens = zip(*sorted(zip(tokens_importances, tokens), reverse=True))

    print(tokens[:n])
    print(tokens_importances[:n])


def setup_logger():
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s')
    log.setLevel(logging.INFO)


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1

    return reviewFeatureVecs


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

    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train_vect.toarray(), y_train)
    print_score_model(model, X_test_vect.toarray(), y_test)

    token_importances = feature_importances(X_train_vect.toarray(), y_train)

    print_most_valuable_features(tokenize_documents_extend(X_train), token_importances)

    # Test Features char ngram
    vectorizer = TfidfVectorizer(max_features=50000, min_df=5, analyzer='char', ngram_range=(3, 3))
    X_train_vect_word_ngram = vectorizer.fit_transform(X_train)
    X_test_vect_word_ngram = vectorizer.transform(X_test)

    train_matrix = np.append(X_train_vect.toarray(), X_train_vect_word_ngram.toarray(), axis=1)
    test_matrix = np.append(X_test_vect.toarray(), X_test_vect_word_ngram.toarray(), axis=1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(train_matrix, y_train)
    print_score_model(model, test_matrix, y_test)

    # Test Features word ngram
    vectorizer = TfidfVectorizer(max_features=50000, min_df=5, tokenizer=tokenize, analyzer='word', ngram_range=(3, 3))
    X_train_vect_word_ngram = vectorizer.fit_transform(X_train)
    X_test_vect_word_ngram = vectorizer.transform(X_test)

    train_matrix = np.append(X_train_vect.toarray(), X_train_vect_word_ngram.toarray(), axis=1)
    test_matrix = np.append(X_test_vect.toarray(), X_test_vect_word_ngram.toarray(), axis=1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(train_matrix, y_train)
    print_score_model(model, test_matrix, y_test)

    # Test Features avg Word2Vec
    train_sentences = tokenize_documents_append(X_train)
    word2vec_train_model = Word2Vec(train_sentences)
    train_avg_vectors = getAvgFeatureVecs(train_sentences, word2vec_train_model, 100)

    test_sentences = tokenize_documents_append(X_test)
    word2vec_test_model = Word2Vec(test_sentences)
    test_avg_vectors = getAvgFeatureVecs(test_sentences, word2vec_test_model, 100)

    X_train_bow_vectors = np.append(X_train_vect.toarray(), train_avg_vectors, axis=1)
    X_test_bow_vectors = np.append(X_test_vect.toarray(), test_avg_vectors, axis=1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train_bow_vectors, y_train)
    print_score_model(model, X_test_bow_vectors, y_test)

    # Test Feature word count
    train_sentences = tokenize_documents_append(X_train)
    train_word_counts = [len(tokens) for tokens in train_sentences]
    train_word_counts = np.reshape(train_word_counts, (-1, 1))

    test_sentences = tokenize_documents_append(X_test)
    test_word_counts = [len(tokens) for tokens in test_sentences]
    test_word_counts = np.reshape(test_word_counts, (-1, 1))

    train_matrix = np.append(X_train_vect.toarray(), train_word_counts, axis=1)
    test_matrix = np.append(X_test_vect.toarray(), test_word_counts, axis=1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(train_matrix, y_train)
    print_score_model(model, test_matrix, y_test)

    # Test Feature characters count
    train_char_counts = [len(document) for document in X_train]
    train_char_counts = np.reshape(train_char_counts, (-1, 1))

    test_char_counts = [len(document) for document in X_test]
    test_char_counts = np.reshape(test_char_counts, (-1, 1))

    train_matrix = np.append(X_train_vect.toarray(), train_char_counts, axis=1)
    test_matrix = np.append(X_test_vect.toarray(), test_char_counts, axis=1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(train_matrix, y_train)
    print_score_model(model, test_matrix, y_test)

    # Test Feature brackets count
    train_bracket_counts = [document.count(')') - document.count('(') for document in X_train]
    train_bracket_counts = np.reshape(train_bracket_counts, (-1, 1))

    test_bracket_counts = [document.count(')') - document.count('(') for document in X_test]
    test_bracket_counts = np.reshape(test_bracket_counts, (-1, 1))

    train_matrix = np.append(X_train_vect.toarray(), train_bracket_counts, axis=1)
    test_matrix = np.append(X_test_vect.toarray(), test_bracket_counts, axis=1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(train_matrix, y_train)
    print_score_model(model, test_matrix, y_test)
