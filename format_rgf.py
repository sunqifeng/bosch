import pandas
import numpy

import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics

def format_data(numeric_path, categorical_path, date_path, vectorizer, chunksize=200000):

    train_numeric = pandas.read_csv(numeric_path, chunksize=chunksize, dtype=numpy.float32)

    #train_categorical = pandas.read_csv(categorical_path, chunksize=chunksize)
    #train_categorical = vectorizer.fit(train_categorical, None)

    train_date = pandas.read_csv(date_path, chunksize=chunksize, dtype=numpy.float32)

    reader = zip(train_numeric, train_date)

    first = True
    for numeric, date in reader:
        numeric.drop('Id', axis=1, inplace=True)
        #categorical.drop('Id', axis=1, inplace=True)
        date.drop('Id', axis=1, inplace=True)

        chunk_data = pandas.concat([numeric, date], axis=1)
        chunk_data.fillna(0.0)
        positive = chunk_data[chunk_data['Response'] == 1]
        negative = chunk_data[chunk_data['Response'] == 0].sample(frac=0.1)
        chunk_data = pandas.concat([positive, negative])
        if first:
            data = chunk_data.copy()
            first = False
        else:
            data = pandas.concat([data, chunk_data])
        print(chunk_data.shape, data.shape)
    return data

train_data = format_data("./data/train_numeric.csv", "./data/train_categorical.csv", "./data/train_date.csv", vectorizer)