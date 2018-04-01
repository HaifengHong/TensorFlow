# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###### 2-1创建图，启动图
# # 创建常量op
# n1 = tf.constant([[3,3]])
# n2 = tf.constant([[2],[3]])
# # 创建一个矩阵乘法op，把n1、n2传入
# product = tf.matmul(n1, n2)
# print(product)
# # 定义一个会话，启动默认图
# with tf.Session() as sess:
#     # 调用sess的run方法来执行矩阵乘法op，run(product)触发了图中3个op
#     result = sess.run(product)
#     print(result)

###### 2-2变量
# x = tf.Variable([1,2])
# a = tf.constant([3,3])
# # 增加一个减法op
# sub = tf.subtract(x,a)
# # 增加一个加法op
# add = tf.add(x,sub)
# # 初始化所有变量
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))

# # 创建一个变量，初始化为0
# state = tf.Variable(0, name='counter')
# # 创建一个加法op，作用是使state加1
# new_value = tf.add(state, 1)
# # 赋值op
# update = tf.assign(state, new_value)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print('初始值：')
#     print(sess.run(state))
#     print('循环开始：')
#     for _ in range(5):
#         sess.run(update)
#         print(sess.run(state))

####### 2-3Fetch and Feed
# # Fetch 对多个操作节点取值
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# add = tf.add(input2,input3)
# mul = tf.multiply(input1,add)
# with tf.Session() as sess:
#     result = sess.run([mul,add]) # 运行多个op
#     print(result)
# # Feed 给占位符赋值
# # 创建占位符
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run(output,feed_dict={input1:7.,input2:2.}))

##### tensorflow简单示例
# 使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2
# 构造一个线性模型
k = tf.Variable(0.) # 初始值，浮点型
b = tf.Variable(0.)
y = k * x_data + b
# 定义代价函数
loss = 0.5 * tf.reduce_mean(tf.square(y_data - y)) # 误差的平方再平均
# 定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2) # 学习率0.2
# 定义一个训练，最小化代价函数
train = optimizer.minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201): # 迭代200次
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k,b])) # 每迭代20次看一下结果
# 最终结果应该接近于k=0.1,b=0.2