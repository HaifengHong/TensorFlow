# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # NOT import ... as ....
from tensorflow.contrib.tensorboard.plugins import projector  # NOT import ... as ...

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 运行次数
max_step = 1001

# 图片数量
image_num = 3000  # PROJECTOR窗口（旧TF版本叫EMBEDDINGS）中显示的图片数量

# 文件路径
# DIR = 'D:\PyCharmCommunityEdition2017.2.4\PyTests\TensorFlow' \改成/，最后加一个/
DIR = 'D:/PyCharmCommunityEdition2017.2.4/PyTests/TensorFlow/'

# 定义会话
sess = tf.Session()

# 载入图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summary'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')  # NOT x = tf.placeholder('tf.float32', ...)
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

# 在IMAGE窗口显示图片
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])  # -1自动计算大小；用的是黑白图片所以这里维度是1(若是彩色，则为3)
    tf.summary.image('input', image_shaped_input, 10)  # 放10张图片

# 构建神经网络
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10], tf.float32, name='W'))  # NOT W = tf.zeros([784, 500], tf.float32)
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), tf.float32, name='b')  # NOT b = tf.Variable(tf.zeros(10, 10)...)
        variable_summaries(b)
    with tf.name_scope('prediction'):
        prediction = tf.matmul(x, W) + b

with tf.name_scope('loss'):
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 产生metadata文件
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):  # 检查有无这个文件存在，若有则删掉
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')  # 若运行出现错误，把该路径下的文件删掉即可
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))  # 将标签代表的实际数字赋给labels
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')  # metadata.tsv文件中将会有image_num个换行展示的label

# 合并所有summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
saver = tf.train.Saver()  # saver用来保存网络模型（程序导数第三行用到）
config = projector.ProjectorConfig()  # 定义配置项
embed = config.embeddings.add()  # embedding在前面定义过
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'  # 给出metadata文件路径
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'  # 给出图片路径
embed.sprite.single_image_dim.extend([28, 28])  # 将mnist_10k_sprite.png这个图片按照28*28像素进行切分
projector.visualize_embeddings(projector_writer, config)

for i in range(max_step):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)  # 固定搭配
    run_metadata = tf.RunMetadata()  # 固定搭配
    summary, _ = sess.run([merged, train], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                          run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)

    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Iter', i, 'Testing Accuracy =', acc)

saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_step)  # 将训练好的模型保存到这个路径
projector_writer.close()
sess.close()


# D:\Python3.6.4\python.exe D:/PyCharmCommunityEdition2017.2.4/PyTests/TensorFlow/YTTutorial5-4_Tensorboard_Embedding.py
# Extracting MNIST_data\train-images-idx3-ubyte.gz
# Extracting MNIST_data\train-labels-idx1-ubyte.gz
# Extracting MNIST_data\t10k-images-idx3-ubyte.gz
# Extracting MNIST_data\t10k-labels-idx1-ubyte.gz
# 2018-05-20 19:12:14.444325: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# 2018-05-20 19:12:14.997581: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties:
# name: GeForce 840M major: 5 minor: 0 memoryClockRate(GHz): 1.124
# pciBusID: 0000:01:00.0
# totalMemory: 2.00GiB freeMemory: 1.66GiB
# 2018-05-20 19:12:14.997927: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0
# 2018-05-20 19:12:15.870637: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1429 MB memory) -> physical GPU (device: 0, name: GeForce 840M, pci bus id: 0000:01:00.0, compute capability: 5.0)
# 2018-05-20 19:12:17.538117: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\stream_executor\dso_loader.cc:151] successfully opened CUDA library cupti64_90.dll locally
# Iter 0 Testing Accuracy = 0.1425
# Iter 100 Testing Accuracy = 0.8954
# Iter 200 Testing Accuracy = 0.9051
# Iter 300 Testing Accuracy = 0.9104
# Iter 400 Testing Accuracy = 0.9093
# Iter 500 Testing Accuracy = 0.9154
# Iter 600 Testing Accuracy = 0.914
# Iter 700 Testing Accuracy = 0.9155
# Iter 800 Testing Accuracy = 0.9196
# Iter 900 Testing Accuracy = 0.9193
# Iter 1000 Testing Accuracy = 0.9207
#
# Process finished with exit code 0