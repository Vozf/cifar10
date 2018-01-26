from input import load_training_data, load_test_data, maybe_download_and_extract
import tensorflow as tf
import sys

MODEL_STORE_PATH = './model/v1/dump/my-model'
SUMMARY_STORE_PATH = './model/v1/summary'


def create_nn(x):
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([8 * 8 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 1024])
        b_fc2 = bias_variable([1024])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('out'):
        W_fc3 = weight_variable([1024, 10])
        b_fc3 = bias_variable([10])

        out = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    return out, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_input_fn(batch_size, test_size):
    (training_data, _, training_labels) = load_training_data()
    (test_data, _, test_labels) = load_test_data()

    training_data = tf.convert_to_tensor(training_data, tf.float32)
    test_data = tf.convert_to_tensor(test_data, tf.float32)

    training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

    training_dataset = training_dataset.shuffle(1000).repeat().batch(batch_size)
    test_dataset = test_dataset.take(test_size).batch(test_size)
    # test_dataset = test_dataset.batch(test_data.shape.dims[0].value)


    # Build the Iterator, and return the read end of the pipeline.
    return training_dataset.make_one_shot_iterator().get_next(), \
           test_dataset.make_one_shot_iterator().get_next()


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def get_train_step(y_, logits, global_step):
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)
        return train_step


def get_accuracy(y_, logits):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def training_session(sess, train_fn, test_fn, global_step, train_step, x, y_, keep_prob,
                     accuracy):
    train_writer = tf.summary.FileWriter(SUMMARY_STORE_PATH + '/train')
    test_writer = tf.summary.FileWriter(SUMMARY_STORE_PATH + '/test')

    merged = tf.summary.merge_all()

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)


    test_data_evaled, test_labels_evaled = sess.run(test_fn)

    saver = tf.train.Saver()
    try:
        saver.restore(sess, MODEL_STORE_PATH)
    except:
        pass

    while True:
        bat = sess.run(train_fn)
        train_step.run(feed_dict={x: bat[0], y_: bat[1], keep_prob: 0.5})

        i = sess.run(global_step)

        if i % 100 == 0:
            summary_test, acc_test = \
                sess.run([merged, accuracy],
                         feed_dict={x: test_data_evaled, y_: test_labels_evaled, keep_prob: 1})

            test_writer.add_summary(summary_test, i)

            summary_train, acc_train = \
                sess.run([merged, accuracy], feed_dict={x: bat[0], y_: bat[1], keep_prob: 1})

            train_writer.add_summary(summary_train, i)

            print('step %d, training accuracy %g, test accuracy %g' % (i, acc_train, acc_test))

        if i % 1000 == 0:
            saver.save(sess, MODEL_STORE_PATH)
            print('model saved')


def finalize_session(coord, threads):
    print('finalize')
    coord.request_stop()
    coord.join(threads)
    sys.exit()


def _main():
    maybe_download_and_extract()

    train_fn, test_fn = get_input_fn(64, 1000)

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, [None, 10])

    global_step = tf.Variable(0, name='global_step', trainable=False)

    logits, keep_prob = create_nn(x)

    train_step = get_train_step(y_, logits, global_step)
    accuracy = get_accuracy(y_, logits)


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        try:
            training_session(sess, train_fn, test_fn, global_step, train_step, x, y_, keep_prob,
                             accuracy)
        except KeyboardInterrupt:
            finalize_session(coord, threads)


if __name__ == '__main__':
    _main()
