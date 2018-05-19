# -*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size


# 定义一个函数，参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)  # 使用到上面定义的函数（一般是分析权值和偏置值）
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)  # 使用到上面定义的函数
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    prediction = wx_plus_b

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)  # 不像W和b，loss只有一个数，没必要调用上面的函数

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()  # 不需取名字，本身就有个名字叫init

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)  # 观察准确率的变化

# 合并所有的summary
merged = tf.summary.merge_all()  # 注意不是merged_all

with tf.Session() as sess:
    sess.run(init)
    write = tf.summary.FileWriter('loginputlayersummary/', sess.graph)  # 在当前路径下的log里写入文件（若没有log，则自动创建），文件就是graph的结构
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs,
                                                                   y: batch_ys})  # 每一次迭代都计算一次merged和train_step，merged的返回值存到summary里（其中的accuracy是训练集准确率，不是输出结果的测试集准确率）

        write.add_summary(summary, epoch)  # 把summary记录下来写到生成的文件里
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}) # 测试集的准确率
        print('Iter ', epoch, ',Testing Accuracy ', acc)


# D:\Python3.6.4\python.exe D:/PyCharmCommunityEdition2017.2.4/PyTests/TensorFlow/YTTutorial5-3_TensorBoard_InputLayer_Scalar.py
# Extracting MNIST_data\train-images-idx3-ubyte.gz
# Extracting MNIST_data\train-labels-idx1-ubyte.gz
# Extracting MNIST_data\t10k-images-idx3-ubyte.gz
# Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
# WARNING:tensorflow:From D:/PyCharmCommunityEdition2017.2.4/PyTests/TensorFlow/YTTutorial5-3_TensorBoard_InputLayer_Scalar.py:42: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
#
# Future major versions of TensorFlow will allow gradients to flow
# into the labels input on backprop by default.
#
# See tf.nn.softmax_cross_entropy_with_logits_v2.
#
# 2018-05-19 20:23:29.462993: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# 2018-05-19 20:23:30.184731: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties:
# name: GeForce 840M major: 5 minor: 0 memoryClockRate(GHz): 1.124
# pciBusID: 0000:01:00.0
# totalMemory: 2.00GiB freeMemory: 1.66GiB
# 2018-05-19 20:23:30.185098: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0
# 2018-05-19 20:23:32.511598: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1429 MB memory) -> physical GPU (device: 0, name: GeForce 840M, pci bus id: 0000:01:00.0, compute capability: 5.0)
# Iter  0 ,Testing Accuracy  0.91
# Iter  1 ,Testing Accuracy  0.9146
# Iter  2 ,Testing Accuracy  0.9181
# Iter  3 ,Testing Accuracy  0.9217
# Iter  4 ,Testing Accuracy  0.921
# Iter  5 ,Testing Accuracy  0.9228
# Iter  6 ,Testing Accuracy  0.9231
# Iter  7 ,Testing Accuracy  0.9223
# Iter  8 ,Testing Accuracy  0.9214
# Iter  9 ,Testing Accuracy  0.9215
# Iter  10 ,Testing Accuracy  0.9209
# Iter  11 ,Testing Accuracy  0.9233
# Iter  12 ,Testing Accuracy  0.925
# Iter  13 ,Testing Accuracy  0.922
# Iter  14 ,Testing Accuracy  0.9202
# Iter  15 ,Testing Accuracy  0.9251
# Iter  16 ,Testing Accuracy  0.9233
# Iter  17 ,Testing Accuracy  0.9243
# Iter  18 ,Testing Accuracy  0.9233
# Iter  19 ,Testing Accuracy  0.9251
# Iter  20 ,Testing Accuracy  0.9255
# Iter  21 ,Testing Accuracy  0.9255
# Iter  22 ,Testing Accuracy  0.9227
# Iter  23 ,Testing Accuracy  0.9238
# Iter  24 ,Testing Accuracy  0.9251
# Iter  25 ,Testing Accuracy  0.9246
# Iter  26 ,Testing Accuracy  0.9261
# Iter  27 ,Testing Accuracy  0.9236
# Iter  28 ,Testing Accuracy  0.9265
# Iter  29 ,Testing Accuracy  0.9242
# Iter  30 ,Testing Accuracy  0.9229
#
# Process finished with exit code 0