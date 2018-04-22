# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#####MNIST数据集分类简单版本(手写数字识别)

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 会自动下载并以“MNIST_data”为文件名保存到当前目录

# 设定每个批次的大小，每次训练用batchsize个样本，矩阵形式
batch_size = 100

# 计算一共多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])

# 创建一个简单的没有隐藏层的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

# softmax交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

# 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 使用AdamOptimizer（学习率一般设置得比较小）
train_step = tf.train.AdamOptimizer(1e-2).minimize(loss) # 学习率1e-2即0.01

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
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter ' + str(epoch) + ',Testing Accuracy ' + str(acc))

# 学习率为1e-4时的结果：
# Iter 0,Testing Accuracy 0.7586
# Iter 1,Testing Accuracy 0.7948
# Iter 2,Testing Accuracy 0.8559
# Iter 3,Testing Accuracy 0.8714
# Iter 4,Testing Accuracy 0.8816
# Iter 5,Testing Accuracy 0.8881
# Iter 6,Testing Accuracy 0.8938
# Iter 7,Testing Accuracy 0.898
# Iter 8,Testing Accuracy 0.9004
# Iter 9,Testing Accuracy 0.903
# Iter 10,Testing Accuracy 0.9053
# Iter 11,Testing Accuracy 0.9074
# Iter 12,Testing Accuracy 0.9091
# Iter 13,Testing Accuracy 0.9098
# Iter 14,Testing Accuracy 0.9108
# Iter 15,Testing Accuracy 0.9111
# Iter 16,Testing Accuracy 0.9122
# Iter 17,Testing Accuracy 0.9133
# Iter 18,Testing Accuracy 0.9149
# Iter 19,Testing Accuracy 0.9159
# Iter 20,Testing Accuracy 0.9163

# 学习率为1e-2时的结果（比1e-4效果好）：
# Iter 0,Testing Accuracy 0.9258
# Iter 1,Testing Accuracy 0.9259
# Iter 2,Testing Accuracy 0.9273
# Iter 3,Testing Accuracy 0.9302
# Iter 4,Testing Accuracy 0.9288
# Iter 5,Testing Accuracy 0.9295
# Iter 6,Testing Accuracy 0.9286
# Iter 7,Testing Accuracy 0.9292
# Iter 8,Testing Accuracy 0.9282
# Iter 9,Testing Accuracy 0.9321
# Iter 10,Testing Accuracy 0.9274
# Iter 11,Testing Accuracy 0.9307
# Iter 12,Testing Accuracy 0.9302
# Iter 13,Testing Accuracy 0.9304
# Iter 14,Testing Accuracy 0.9297
# Iter 15,Testing Accuracy 0.9325
# Iter 16,Testing Accuracy 0.9308
# Iter 17,Testing Accuracy 0.9303
# Iter 18,Testing Accuracy 0.9321
# Iter 19,Testing Accuracy 0.93
# Iter 20,Testing Accuracy 0.9322