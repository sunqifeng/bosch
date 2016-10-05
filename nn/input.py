
import tempfile

import tensorflow as tf
from sklearn import cross_validation

def read_format(filename_queue):
    reader = tf.TextLineReader()
    _, record_string = reader.read(filename_queue)
    record_defaults = [['null'] * 2140, [0.0] * 1156, [0.0] * 969]
    data = tf.decode_csv(record_string, record_defaults=record_defaults)
    categorical = data[0:2140]
    date = data[2140:2140 + 1156]
    numerics = data[2140 + 1156: -1]
    response = data[-1]
    return categorical, date, numerics, response


def input_pipeline(filenames, batch_size, num_epochs=None):
  numerics_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=False)

  categorical, date, numerics, response = read_format(numerics_queue)

  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  categorical_batch, date_batch, numerics_batch, response_batch = tf.train.shuffle_batch(
      [categorical, date, numerics, response], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return categorical_batch, date_batch, numerics_batch, response_batch



def build_estimator(model_dir):
    date = tf.contrib.layers.real_valued_column('date', dimension=1156)
    numerics = tf.contrib.layers.real_valued_column('numerics', dimension=969)
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=[date, numerics],
                                       hidden_units=[1000, 500, 100])

    return m

flags = tf.app.flags
FLAGS = flags.FLAGS

model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir

m = build_estimator(model_dir)

m.fit(input_fn=input_pipeline(['./data/train.merge.csv'], 32), steps=200)

pred_p = m.predict_proba(input_pipeline(['./data/train.merge.csv'], 32), steps=1)


