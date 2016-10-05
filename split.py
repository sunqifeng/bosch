

import numpy
import pandas

from sklearn.cross_validation import train_test_split

def load_data(numeric_path, categorical_path, date_path, chunksize=200000):

    train_numeric = pandas.read_csv(numeric_path, chunksize=chunksize, index_col='Id', dtype=numpy.float32)

    train_categorical = pandas.read_csv(categorical_path, chunksize=chunksize, index_col='Id')
    #train_categorical = vectorizer.fit(train_categorical, None)

    train_date = pandas.read_csv(date_path, chunksize=chunksize, index_col='Id', dtype=numpy.float32)

    reader = zip(train_numeric, train_date)

    first = True
    with open(numeric_path + ".train", "rb") as numeric_train, \
        open(categorical_path + ".train", "rb") as categorical_train, \
        open(date_path + ".train", "rb") as date_train, \
        open(numeric_path + ".valid", "rb") as numeric_valid, \
        open(categorical_path + ".valid", "rb") as categorical_valid, \
        open(date_path + ".valid", "rb") as date_valid:

        for numerics, categorical, date in reader:
            train_numerics, valid_train_numerics, train_categorical, valid_categorical, train_date, valid_date = \
                train_test_split(numerics, categorical, date, train_size=0.7)
            train_numerics.to_csv()
    return data