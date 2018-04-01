# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#####非线性回归

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis] # 增加一维，变成200行1列的数组
print('x_data的形状：', x_data.shape)
noise = np.random.normal(0,0.02,x_data.shape) # 正态分布
y_data = np.square(x_data) + noise # 本来x_data、y_data应该是平方关系（呈U型），但加入noise之后这些点会稍微上下移动一些
# 定义两个placeholder
x = tf.placeholder(tf.float32,[None,1]) # x的shape是根据样本x_data的shape来定义的，None可以是任意值，上面是200则这里的None接收200，注意用list[]表示shape
y = tf.placeholder(tf.float32,[None,1])
# 定义神经网络中间层，L1是中间层的输出
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10])) # 这1个输入层的偏置分别输入到中间层的神经元，而中间层有10个神经元，所以是[1,10]
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1) # 用tanh当作激活函数（为何用tanh？）
# 定义神经网络输出层，prediction是输出层的输出，即预测值
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1])) # 这1个中间层的偏置分别输入到输出层的神经元，而输出层只有1个神经元，所以是[1,1]
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)
# 定义二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict = {x:x_data, y:y_data})
    #获得预测值
    prediction_value = sess.run(prediction, feed_dict = {x:x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5) # 红色实线，线宽5
    plt.show()