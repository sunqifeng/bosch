

import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import mnist
from config import config

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('train_dir', '/tmp/data',
                    'Directory with the training data.')

def read(file_queue):
  reader = tf.TFRecordReader()
  _, example = reader.read(file_queue)
  features = tf.parse_single_example(
      example,
      # Defaults are not specified since both keys are required.
      features={
          'numeric': tf.FixedLenFeature([], tf.float32),
          'categorical': tf.FixedLenFeature([], tf.string),
          'date': tf.FixedLenFeature([], tf.float32),
          'Response': tf.FixedLenFeature([], tf.int64),
      })
  numeric = features['numeric']
  categorical = features['categorical']
  date = features['date']
  response = features['Response']

  return numeric, categorical, date, response


def inputs(train, batch_size=32, num_epochs=10):

  with tf.name_scopes('input'):
    filename_queue = tf.train.string_input_producer(
        [config.train_record], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    numeric, categorical, date, response = read(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    numeric, categorical, date, response = tf.train.shuffle_batch(
        [numeric, categorical, date, response], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return numeric, categorical, date, response



def train():
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # features and labels.
    numeric, categorical, date, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)

    # Build a Graph that computes predictions from the inference model.
    learn.LinearRegressor(feature_columns=learn.read_batch_examples(numeric))

    # Add to the Graph the loss calculation.
    loss = mnist.loss(logits, labels)

    # Add to the Graph operations that train the model.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # The op for initializing the variables.
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())

    # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()

        # Run one step of the model.  The return values are
        # the activations from the `train_op` (which is
        # discarded) and the `loss` op.  To inspect the values
        # of your ops or variables, you may include them in
        # the list passed to sess.run() and the value tensors
        # will be returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss])

        duration = time.time() - start_time

        # Print an overview fairly often.
        if step % 100 == 0:
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                     duration))
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


