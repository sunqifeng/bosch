

import pandas
import numpy

import dask.dataframe as dd

import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
import cPickle
from sklearn.metrics import matthews_corrcoef

def load_data(numeric_path, categorical_path, date_path, chunksize=100000):

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
        #chunk_data.fillna(0.0)
        #positive = chunk_data[chunk_data['Response'] == 1]
        #negative = chunk_data[chunk_data['Response'] == 0].sample(frac=0.5)
        #chunk_data = pandas.concat([positive, negative])
        if first:
            data = dd.from_pandas(chunk_data, chunksize=chunksize)
            first = False
        else:
            data = dd.concat([data, chunk_data], axis=0, interleave_partitions=True)
        #print(chunk_data.shape, data.shape)
    return data

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / numpy.sqrt(inf)

def eval_mcc(y_true, y_prob, show=False):
    idx = numpy.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * numpy.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = numpy.zeros(n)
    for i in range(n):
        # all items with idx <= i are predicted negative while others are predicted positive
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc


def find_matthews_threshold(y_valid, p_valid, try_all=False, verbose=False):
    p_valid, y_valid = numpy.array(p_valid), numpy.array(y_valid)

    best = 0
    best_score = -2
    totry = numpy.arange(0,1,0.01) if try_all is False else numpy.unique(p_valid)
    for t in totry:
        score = matthews_corrcoef(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True:
        print('Best score: ', round(best_score, 5), ' @ threshold ', best)

    return best

def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc

data = load_data("./data/train_numeric.csv", "./data/train_categorical.csv", "./data/train_date.csv")

from sklearn.cross_validation import train_test_split
#train, valid = train_test_split(data, train_size=0.8)
train, valid = data.random_split([0.8, 0.2])
train_y = train['Response']
train = train.drop('Response', axis=1)

valid_y = valid['Response']
valid = valid.drop('Response', axis=1)

import xgboost
from sklearn import grid_search

pos_size = train_y.sum()
neg_size = len(train_y) - pos_size
print(['positive/negative=%d/%d\n' % (pos_size, neg_size)])

xgb_classifier = xgboost.XGBClassifier(silent=False, scale_pos_weight=neg_size/pos_size, base_score=0.005)


xgb_classifier = grid_search.GridSearchCV(xgb_classifier,
                                         param_grid={'max_depth': [4],
                                                     # 'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5],
                                                     # 'reg_lambda': [0.7, 0.8, 0.9]
                                                     })

xgb_classifier.fit(train, train_y)
valid_prob = xgb_classifier.predict_proba(valid)[:, 1]

find_matthews_threshold(valid_y, valid_prob, try_all=True, verbose=True)

#best_classifier = xgb_classfier.best_estimator_
#print(xgb_classifier.feature_importances_[xgb_classifier.feature_importances_ > 0.005])

with open("xgb_model.cpk", "wb") as output:
    cPickle.dump(xgb_classifier.best_estimator_, output)

'''
test_data = load_data("./data/test_numeric.csv", "./data/test_categorical.csv", "./data/test_date.csv")


pred_test_y = xgb_classifier.predict(test_data)

sub = pandas.read_csv("./data/sample_submission.csv")
sub['Response'] = pred_test_y
sub.to_csv("submission.csv")
'''



