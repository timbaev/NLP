import ast
import logging
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize

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
    stemmer = SnowballStemmer("russian")

    tokens = word_tokenize(document, language='russian')

    tokens = [w.lower() for w in tokens if w not in ignore]
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [stemmer.stem(w) for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]

    return tokens


def print_score_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = precision_recall_fscore_support(y_pred=y_pred, y_true=y_test, average='binary', pos_label=POS_MARK)
    log.info(f'Precision: {round(metrics[0], 3)}, recall: {round(metrics[1], 3)}, f-measure: {round(metrics[2], 3)}')


def setup_logger():
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s')
    log.setLevel(logging.INFO)


if __name__ == '__main__':
    setup_logger()

    reviews, marks = read_data(CORPUS_PATH)
    pos_reviews, neg_reviews = split_reviews(reviews, marks)

    y_pos = [1] * len(pos_reviews)
    y_neg = [0] * len(neg_reviews)

    X_train, X_test, y_train, y_test = train_test_split(pos_reviews + neg_reviews, y_pos + y_neg, test_size=0.2)

    # ToDo Это так работает с tokenizer?
    vectorizer = TfidfVectorizer(max_features=50000, min_df=5, tokenizer=tokenize)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train_vect.toarray(), y_train)
    print_score_model(model, X_test_vect.toarray(), y_test)

