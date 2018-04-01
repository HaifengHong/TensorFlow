# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#####MNIST数据集分类简单版本(手写数字识别)

# 载入数据集
# 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。
# one_hot表示用非零即1的数组保存图片表示的数值.比如一个图片上面写的是1,那个保存的就是[0,1,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 会自动下载并以“MNIST_data”为文件名保存到当前目录
# 设定每个批次的大小，每次训练用batchsize个样本，矩阵形式
batch_size = 100 #batch_size取值越大，训练结果（准确率）越差
# 计算一共多少个批次
n_batch = mnist.train.num_examples // batch_size
# 定义两个placeholder
x = tf.placeholder(tf.float32, [None,784]) # None表示此张量的第一个维度可以是任何长度的，这里None等于batch_size的大小即100，784表示28像素*28像素=784
y = tf.placeholder(tf.float32, [None,10]) # 标签是0~9，所以是10
# 创建一个简单的没有隐藏层的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10])) # 1行10列
prediction = tf.nn.softmax(tf.matmul(x,W)+b) # 用softmax作激活函数
# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# softmax交叉熵代价函数，作对比
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction)) # 返回的是个向量，注意求平均才能得到loss（根据交叉熵代价函数公式）。疑问：用tf.reduce_sum为何结果很差？
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 结果存放在布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) # tf.argmax返回一维张量中最大值所在的位置；tf.equal返回True或False
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 将布尔型转化为浮点型，True/False转化成1.0/0.0，再求平均值即得准确率
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21): # 把所有图片训练21次
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}) # 求每次epoch的准确率，用测试集的图片和标签
        print('Iter ' + str(epoch) + ',Testing Accuracy ' + str(acc))

# 用二次代价函数预测结果：
# Iter 0,Testing Accuracy 0.831
# Iter 1,Testing Accuracy 0.8706
# Iter 2,Testing Accuracy 0.8812
# Iter 3,Testing Accuracy 0.8883
# Iter 4,Testing Accuracy 0.8943
# Iter 5,Testing Accuracy 0.8976
# Iter 6,Testing Accuracy 0.8991
# Iter 7,Testing Accuracy 0.9012
# Iter 8,Testing Accuracy 0.9035
# Iter 9,Testing Accuracy 0.9054
# Iter 10,Testing Accuracy 0.9067
# Iter 11,Testing Accuracy 0.9073
# Iter 12,Testing Accuracy 0.9081
# Iter 13,Testing Accuracy 0.9089
# Iter 14,Testing Accuracy 0.9093
# Iter 15,Testing Accuracy 0.9104
# Iter 16,Testing Accuracy 0.9118
# Iter 17,Testing Accuracy 0.9121
# Iter 18,Testing Accuracy 0.9124
# Iter 19,Testing Accuracy 0.9134
# Iter 20,Testing Accuracy 0.9137

# 用softmax交叉熵代价函数预测结果（对比得知，预测结果比二次代价函数要好）：
# Iter 0,Testing Accuracy 0.847
# Iter 1,Testing Accuracy 0.8951
# Iter 2,Testing Accuracy 0.9021
# Iter 3,Testing Accuracy 0.9058
# Iter 4,Testing Accuracy 0.9079
# Iter 5,Testing Accuracy 0.9111
# Iter 6,Testing Accuracy 0.9118
# Iter 7,Testing Accuracy 0.913
# Iter 8,Testing Accuracy 0.9153
# Iter 9,Testing Accuracy 0.9164
# Iter 10,Testing Accuracy 0.9181
# Iter 11,Testing Accuracy 0.9178
# Iter 12,Testing Accuracy 0.9188
# Iter 13,Testing Accuracy 0.9195
# Iter 14,Testing Accuracy 0.9207
# Iter 15,Testing Accuracy 0.9206
# Iter 16,Testing Accuracy 0.9196
# Iter 17,Testing Accuracy 0.9216
# Iter 18,Testing Accuracy 0.9213
# Iter 19,Testing Accuracy 0.9213
# Iter 20,Testing Accuracy 0.9218