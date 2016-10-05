
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Config:
  def __init__(self):
    self.train_numeric = './data/train_numeric.csv'
    self.train_categorical = './data/train_categorical.csv'
    self.train_date = './data/train_date.csv'
    self.train_record = './data/train_record.tf'

    self.test_numeric = './data/test_numeric.csv'
    self.test_categorical = './data/test_categorical.csv'
    self.test_date = './data/test_date.csv'
    self.test_record = './data/test_record.tf'


config = Config()

