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
loss = tf.reduce_mean(tf.square(y-prediction))
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
