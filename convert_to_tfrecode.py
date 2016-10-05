# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

import pandas
import numpy
from config import config

tf.app.flags.DEFINE_integer('validation_fract', 0.3,
                            'fraction of examples to separate from the training '
                            'data for the validation set.')
FLAGS = tf.app.flags.FLAGS

float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
str_feature = lambda v: tf.train.Feature(str_list=tf.train.BytesList(value=v))

def convert_to(numeric_path, categorical_path, date_path, tfrecord_name, chunksize=5000):
  numeric_stream = pandas.read_csv(numeric_path, index_col='Id', chunksize=chunksize, dtype=numpy.float32)

  categorical_stream = pandas.read_csv(categorical_path, index_col='Id', chunksize=chunksize, dtype=numpy.str)

  date_stream = pandas.read_csv(date_path, index_col='Id', chunksize=chunksize, dtype=numpy.float32,)

  writer = tf.python_io.TFRecordWriter(tfrecord_name)
  for numeric_batch, categorical_batch, date_batch in zip(numeric_stream, categorical_stream, date_stream):
    response_batch = numeric_batch['Response']
    numeric_batch = numeric_batch.drop('Response', axis=1, inplace=False)
    #numeric_batch.fillna(-10.0)
    #categorical_batch.fillna('')
    #date_batch.fillna(-100.0)
    for numeric, categorical, date, response in zip(numeric_batch, categorical_batch, date_batch, response_batch):
      #print(" " % numeric)
      example = tf.train.Example(features=tf.train.Features(feature={
        'numeric': float_feature(numeric.tolist()),
        'categorical': str_feature(categorical.tolist()),
        'date': float_feature(date.tolist()),
        'Response': float_feature(response)
      }))
      writer.write(example.SerializeToString())
  writer.close()


def main(argv):
  # Get the data.
  print(config)
  convert_to(config.train_numeric, config.train_categorical, config.train_date, config.train_record)
  #convert_to(config.test_numerics, config.test_categorical, config.test_date, config.test_record)
  # Convert to Examples and write the result to TFRecords.


if __name__ == '__main__':
  tf.app.run()