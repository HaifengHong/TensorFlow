# -*-coding:utf-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size

# 命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None,784], name='x_input')
    y = tf.placeholder(tf.float32, [None,10], name='y_input')

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

prediction = tf.matmul(x,W)+b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,axis=1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    write = tf.summary.FileWriter('loginput/', sess.graph) # 在当前路径下的log里写入文件（若没有log，则自动创建），文件就是graph的结构
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter ', epoch,  ',Testing Accuracy ', acc)