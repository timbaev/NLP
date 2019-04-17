import ast
import logging

from sklearn.model_selection import train_test_split


CORPUS_PATH = 'data/corpus.txt'
log = logging.getLogger()

POS_MARK = 1
NEG_MARK = 0


def read_data(corpus_path):
    with open(f'{corpus_path}', 'r') as f:
        log.info("Reading file...")

        reviews = []
        marks = []

        lines = f.readlines()
        lines = [x.strip() for x in lines]

        for line in lines:
            dictionary = ast.literal_eval(line)
            reviews.append(dictionary['description'])

            recom_author_mark = dictionary['recom_author_mark']
            if recom_author_mark == 'ДА':
                marks.append(POS_MARK)
            elif recom_author_mark == '':
                marks.append(NEG_MARK)
            else:
                raise Exception("Unknown recom_author_mark")

    return reviews, marks


def split_reviews(reviews, marks, n=500):
    pos_reviews = []
    neg_reviews = []

    log.info(f"Split reviews to {n} positive and {n} negative.")

    for index, review in enumerate(reviews):
        if marks[index] == POS_MARK and len(pos_reviews) < n:
            pos_reviews.append(review)
        elif len(neg_reviews) < n:
            neg_reviews.append(review)

        if len(pos_reviews) == n and len(neg_reviews) == n:
            break

    return pos_reviews, neg_reviews


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

