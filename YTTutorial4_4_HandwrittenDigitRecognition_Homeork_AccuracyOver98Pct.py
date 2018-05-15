# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batchsize = 100

n_batch = mnist.train.num_examples // batchsize

# x = tf.placeholder(tf.float32, [784, None])
# y = tf.placeholder(tf.float32, [10, 1])
# keep_prob = tf.placeholder(tf.float32, [1])
# lr = tf.placeholder(tf.float32, [1])
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32, [1]) # Wrong
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

# W1 = tf.truncated_normal([batchsize, 500], stddev=1.0)
# b1 = tf.zeros([1, 500])
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]))
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

# W2 = tf.truncated_normal([500, 300], stddev=1.0)
# b2 = tf.zeros([1, 300])
W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]))
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

# W3 = tf.truncated_normal([300, 10], stddev=1.0)
# b3 = tf.zeros([1, 10])
W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
prediction = tf.matmul(L2_drop, W3) + b3

# loss = tf.nn.softmax_cross_entropy_with_logits(L3, y)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

train = tf.train.AdamOptimizer(lr).minimize(loss)

# bool_result = tf.equal(tf.argmax(L3, 1), tf.argmax(y, 1))
bool_result = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(bool_result, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(51):
        # sess.run(tf.assign(lr, 0.001 * 0.95 ** step))
        sess.run(tf.assign(lr, 0.001 * (0.95 ** step)))
        for batch in range(n_batch):
            # train_acc = sess.run(train, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob:1.0})
            # test_acc = sess.run(train, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:1.0})
            # print('Iteration:', step, 'Trainting Accuracy:', train_acc, 'Testing accuracy:', test_acc)
            batch_xs, batch_ys = mnist.train.next_batch(batchsize)
            sess.run(train, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})

        learning_rate = sess.run(lr)
        # acc = sess.run(train, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        print('Iter:', step, 'Testing Accuracy:', acc, 'Learning Rate:', learning_rate)


# D:\Python3.6.4\python.exe D:/PyCharmCommunityEdition2017.2.4/PyTests/TensorFlow/YTTutorial4_4_HandwrittenDigitRecognition_Homeork_AccuracyOver98Pct.py
# Extracting MNIST_data\train-images-idx3-ubyte.gz
# Extracting MNIST_data\train-labels-idx1-ubyte.gz
# Extracting MNIST_data\t10k-images-idx3-ubyte.gz
# Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
# WARNING:tensorflow:From D:/PyCharmCommunityEdition2017.2.4/PyTests/TensorFlow/YTTutorial4_4_HandwrittenDigitRecognition_Homeork_AccuracyOver98Pct.py:43: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
#
# Future major versions of TensorFlow will allow gradients to flow
# into the labels input on backprop by default.
#
# See tf.nn.softmax_cross_entropy_with_logits_v2.
#
# 2018-05-10 23:50:42.937809: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# 2018-05-10 23:50:43.458454: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties:
# name: GeForce 840M major: 5 minor: 0 memoryClockRate(GHz): 1.124
# pciBusID: 0000:01:00.0
# totalMemory: 2.00GiB freeMemory: 1.66GiB
# 2018-05-10 23:50:43.458833: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0
# 2018-05-10 23:50:44.151210: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1429 MB memory) -> physical GPU (device: 0, name: GeForce 840M, pci bus id: 0000:01:00.0, compute capability: 5.0)
# Iter: 0 Testing Accuracy: 0.963 Learning Rate: 0.001
# Iter: 1 Testing Accuracy: 0.9702 Learning Rate: 0.00095
# Iter: 2 Testing Accuracy: 0.9761 Learning Rate: 0.0009025
# Iter: 3 Testing Accuracy: 0.9775 Learning Rate: 0.000857375
# Iter: 4 Testing Accuracy: 0.9774 Learning Rate: 0.00081450626
# Iter: 5 Testing Accuracy: 0.9771 Learning Rate: 0.0007737809
# Iter: 6 Testing Accuracy: 0.9789 Learning Rate: 0.0007350919
# Iter: 7 Testing Accuracy: 0.9803 Learning Rate: 0.0006983373
# Iter: 8 Testing Accuracy: 0.9807 Learning Rate: 0.0006634204
# Iter: 9 Testing Accuracy: 0.9806 Learning Rate: 0.0006302494
# Iter: 10 Testing Accuracy: 0.9812 Learning Rate: 0.0005987369
# Iter: 11 Testing Accuracy: 0.9775 Learning Rate: 0.0005688001
# Iter: 12 Testing Accuracy: 0.9806 Learning Rate: 0.0005403601
# Iter: 13 Testing Accuracy: 0.9824 Learning Rate: 0.0005133421
# Iter: 14 Testing Accuracy: 0.982 Learning Rate: 0.000487675
# Iter: 15 Testing Accuracy: 0.9826 Learning Rate: 0.00046329122
# Iter: 16 Testing Accuracy: 0.9824 Learning Rate: 0.00044012666
# Iter: 17 Testing Accuracy: 0.9824 Learning Rate: 0.00041812033
# Iter: 18 Testing Accuracy: 0.9821 Learning Rate: 0.00039721432
# Iter: 19 Testing Accuracy: 0.9821 Learning Rate: 0.0003773536
# Iter: 20 Testing Accuracy: 0.9824 Learning Rate: 0.00035848594
# Iter: 21 Testing Accuracy: 0.9821 Learning Rate: 0.00034056162
# Iter: 22 Testing Accuracy: 0.9821 Learning Rate: 0.00032353355
# Iter: 23 Testing Accuracy: 0.9821 Learning Rate: 0.00030735688
# Iter: 24 Testing Accuracy: 0.982 Learning Rate: 0.000291989
# Iter: 25 Testing Accuracy: 0.9824 Learning Rate: 0.00027738957
# Iter: 26 Testing Accuracy: 0.9823 Learning Rate: 0.0002635201
# Iter: 27 Testing Accuracy: 0.9822 Learning Rate: 0.00025034408
# Iter: 28 Testing Accuracy: 0.9821 Learning Rate: 0.00023782688
# Iter: 29 Testing Accuracy: 0.9828 Learning Rate: 0.00022593554
# Iter: 30 Testing Accuracy: 0.9827 Learning Rate: 0.00021463877
# Iter: 31 Testing Accuracy: 0.9785 Learning Rate: 0.00020390682
# Iter: 32 Testing Accuracy: 0.9815 Learning Rate: 0.00019371149
# Iter: 33 Testing Accuracy: 0.9813 Learning Rate: 0.0001840259
# Iter: 34 Testing Accuracy: 0.9814 Learning Rate: 0.00017482461
# Iter: 35 Testing Accuracy: 0.9814 Learning Rate: 0.00016608338
# Iter: 36 Testing Accuracy: 0.9814 Learning Rate: 0.00015777921
# Iter: 37 Testing Accuracy: 0.981 Learning Rate: 0.00014989026
# Iter: 38 Testing Accuracy: 0.981 Learning Rate: 0.00014239574
# Iter: 39 Testing Accuracy: 0.9812 Learning Rate: 0.00013527596
# Iter: 40 Testing Accuracy: 0.9815 Learning Rate: 0.00012851215
# Iter: 41 Testing Accuracy: 0.9815 Learning Rate: 0.00012208655
# Iter: 42 Testing Accuracy: 0.9817 Learning Rate: 0.00011598222
# Iter: 43 Testing Accuracy: 0.9817 Learning Rate: 0.00011018311
# Iter: 44 Testing Accuracy: 0.9816 Learning Rate: 0.000104673956
# Iter: 45 Testing Accuracy: 0.9816 Learning Rate: 9.944026e-05
# Iter: 46 Testing Accuracy: 0.9816 Learning Rate: 9.446825e-05
# Iter: 47 Testing Accuracy: 0.9819 Learning Rate: 8.974483e-05
# Iter: 48 Testing Accuracy: 0.9817 Learning Rate: 8.525759e-05
# Iter: 49 Testing Accuracy: 0.9822 Learning Rate: 8.099471e-05
# Iter: 50 Testing Accuracy: 0.9821 Learning Rate: 7.6944976e-05