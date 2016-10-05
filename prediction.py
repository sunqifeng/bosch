

import cPickle
import pandas
import numpy


with open('xgb_model.cpk', 'rb') as file:
    model = cPickle.load(file)


def find_matthews_threshold(y_valid, p_valid, try_all=False, verbose=False):
    p_valid, y_valid = numpy.array(p_valid), numpy.array(y_valid)

    best = 0
    best_score = -2
    totry = numpy.arange(0,1,0.01) if try_all is False else numpy.unique(p_valid)
    for t in totry:
        from sklearn.metrics import matthews_corrcoef
        score = matthews_corrcoef(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True:
        print('Best score: ', round(best_score, 5), ' @ threshold ', best)

    return best


def load_train_data(numeric_path, categorical_path, date_path, chunksize=200000):

    train_numeric = pandas.read_csv(numeric_path, index_col='Id', chunksize=chunksize, dtype=numpy.float32)

    #train_categorical = pandas.read_csv(categorical_path, chunksize=chunksize)
    #train_categorical = vectorizer.fit(train_categorical, None)

    train_date = pandas.read_csv(date_path, index_col='Id', chunksize=chunksize, dtype=numpy.float32)

    reader = zip(train_numeric, train_date)

    first = True
    for numeric, date in reader:
        #numeric.drop('Id', axis=1, inplace=True)
        #categorical.drop('Id', axis=1, inplace=True)
        #date.drop('Id', axis=1, inplace=True)

        chunk_data = pandas.concat([numeric, date], axis=1).sample(frac=0.2)
        chunk_data.fillna(0.0)
        if first:
            data = chunk_data.copy()
            first = False
        else:
            data = pandas.concat([data, chunk_data])
        print(chunk_data.shape, data.shape)
    return data

train_data = load_train_data("./data/train_numeric.csv", "./data/train_categorical.csv", "./data/train_date.csv")
train_y = train_data['Response']
train_data = train_data.drop('Response', axis=1, inplace=False)
prob_y = model.predict_proba(train_data)[:, 1]
threshold = find_matthews_threshold(train_y, prob_y, verbose=True)


def load_test_data(numeric_path, categorical_path, date_path, chunksize=10000):

    train_numeric = pandas.read_csv(numeric_path, index_col='Id', chunksize=chunksize, dtype=numpy.float32)

    #train_categorical = pandas.read_csv(categorical_path, chunksize=chunksize)
    #train_categorical = vectorizer.fit(train_categorical, None)

    train_date = pandas.read_csv(date_path, index_col='Id', chunksize=chunksize, dtype=numpy.float32)
    reader = zip(train_numeric, train_date)

    first = True
    for numeric, date in reader:
        #numeric.drop('Id', axis=1, inplace=True)
        #categorical.drop('Id', axis=1, inplace=True)
        #date.drop('Id', axis=1, inplace=True)

        chunk_data = pandas.concat([numeric, date], axis=1)
        chunk_data.fillna(0.0)
        #positive = chunk_data[chunk_data['Response'] == 1]
        #negative = chunk_data[chunk_data['Response'] == 0].sample(frac=0.5)
        #chunk_data = pandas.concat([positive, negative])
        pred_test_y = model.predict_proba(chunk_data)[:, 1]
        print('\n'.join(['1' if v > threshold else '0' for v in pred_test_y]))

        '''
        if first:
            pred_test = pred_test_y.copy()
            first = False
        else:
            pred_test = pandas.concat([pred_test, pred_test_y])
        print(pred_test_y.shape, pred_test.shape)
        '''
    #return pred_test


load_test_data("./data/test_numeric.csv", "./data/test_categorical.csv", "./data/test_date.csv")
'''
sub = pandas.read_csv("./data/sample_submission.csv")
sub['Response'] = pred_test
sub.to_csv("submission.csv")
'''