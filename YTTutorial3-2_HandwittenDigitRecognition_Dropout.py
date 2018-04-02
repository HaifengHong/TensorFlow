# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#####MNIST数据集分类简单版本(手写数字识别)

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 会自动下载并以“MNIST_data”为文件名保存到当前目录

# 设定每个批次的大小，每次训练用batchsize个样本，矩阵形式
batch_size = 100 # batch_size取值越大，训练结果（准确率）越差
# 计算一共多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义placeholder
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32) # 设置dropout参与工作的神经元比例

# 创建有三个隐藏层的神经网络
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1)) # 权值初始化用截断正态分布效果更好
b1 = tf.Variable(tf.zeros([2000])+0.1) # 偏置初始化还是用0
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_dropout = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1,W2)+b2)
L2_dropout = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2,W3)+b3)
L3_dropout = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_dropout,W4)+b4)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# softmax交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)) # 返回的是个向量，注意求平均才能得到loss（根据交叉熵代价函数公式）。疑问：用tf.reduce_sum为何结果很差？

# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
        Test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels, keep_prob:1.0}) # 用测试集数据测每次epoch的准确率（比训练集高）
        Train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})  # 用训练集数据测每次epoch的准确率
        print('Iter ' + str(epoch) + ',Testing Accuracy ' + str(Test_acc) + ',Training Accuracy ' + str(Train_acc))

# 训练keep_prob=1.0，测试keep_prob=1.0时的结果：
# Iter 0,Testing Accuracy 0.9504,Training Accuracy 0.9596182
# Iter 1,Testing Accuracy 0.9585,Training Accuracy 0.97374547
# Iter 2,Testing Accuracy 0.9602,Training Accuracy 0.9809091
# Iter 3,Testing Accuracy 0.9661,Training Accuracy 0.9854909
# Iter 4,Testing Accuracy 0.9685,Training Accuracy 0.9878182
# Iter 5,Testing Accuracy 0.9675,Training Accuracy 0.9891091
# Iter 6,Testing Accuracy 0.9685,Training Accuracy 0.9901636
# Iter 7,Testing Accuracy 0.9699,Training Accuracy 0.99076366
# Iter 8,Testing Accuracy 0.9704,Training Accuracy 0.9915091
# Iter 9,Testing Accuracy 0.9708,Training Accuracy 0.99201816
# Iter 10,Testing Accuracy 0.9703,Training Accuracy 0.9926909
# Iter 11,Testing Accuracy 0.9701,Training Accuracy 0.9931091
# Iter 12,Testing Accuracy 0.9716,Training Accuracy 0.99349093
# Iter 13,Testing Accuracy 0.9714,Training Accuracy 0.9936727
# Iter 14,Testing Accuracy 0.9717,Training Accuracy 0.9937818
# Iter 15,Testing Accuracy 0.9719,Training Accuracy 0.99398184
# Iter 16,Testing Accuracy 0.9717,Training Accuracy 0.9942
# Iter 17,Testing Accuracy 0.9718,Training Accuracy 0.9943454
# Iter 18,Testing Accuracy 0.9718,Training Accuracy 0.9944364
# Iter 19,Testing Accuracy 0.9717,Training Accuracy 0.9945818
# Iter 20,Testing Accuracy 0.9719,Training Accuracy 0.99472725

# 训练keep_prob=0.7，测试keep_prob=1.0时的结果：
# Iter 0,Testing Accuracy 0.9413,Training Accuracy 0.9459818
# Iter 1,Testing Accuracy 0.9492,Training Accuracy 0.9615091
# Iter 2,Testing Accuracy 0.9591,Training Accuracy 0.9716909
# Iter 3,Testing Accuracy 0.9615,Training Accuracy 0.97725457
# Iter 4,Testing Accuracy 0.9644,Training Accuracy 0.98074543
# Iter 5,Testing Accuracy 0.9667,Training Accuracy 0.9841273
# Iter 6,Testing Accuracy 0.9681,Training Accuracy 0.98627275
# Iter 7,Testing Accuracy 0.9697,Training Accuracy 0.98776364
# Iter 8,Testing Accuracy 0.9689,Training Accuracy 0.9887091
# Iter 9,Testing Accuracy 0.9709,Training Accuracy 0.98996365
# Iter 10,Testing Accuracy 0.971,Training Accuracy 0.9905091
# Iter 11,Testing Accuracy 0.9729,Training Accuracy 0.9911091
# Iter 12,Testing Accuracy 0.9712,Training Accuracy 0.9916
# Iter 13,Testing Accuracy 0.9737,Training Accuracy 0.9921091
# Iter 14,Testing Accuracy 0.9737,Training Accuracy 0.99252725
# Iter 15,Testing Accuracy 0.9736,Training Accuracy 0.9927091
# Iter 16,Testing Accuracy 0.9743,Training Accuracy 0.9931818
# Iter 17,Testing Accuracy 0.9746,Training Accuracy 0.9933636
# Iter 18,Testing Accuracy 0.9752,Training Accuracy 0.9936182
# Iter 19,Testing Accuracy 0.9745,Training Accuracy 0.9939091
# Iter 20,Testing Accuracy 0.9764,Training Accuracy 0.99405456

# 通过对比可知使用dropout收敛速度会变慢。
# 根据训练keep_prob=1.0（未使用dropout）的对比结果，测试集准确率比训练集低了2个百分点，说明出现了过拟合；
# 根据训练keep_prob=0.7（使用dropout）的对比结果，测试集准确率和训练集差不多，说明避免了过拟合。

# 本例看不出是否用dropout对Testing Accuracy的影响，但使用较复杂网络训练较小的训练集时才能看出效果。