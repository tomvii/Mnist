from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("./",one_hot=True)
#print (mnist.train.labels.shape)   #55000,784
#print (mnist.train.images.shape)   #55000,10 one-hot

#softmax

x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder(tf.float32,[None,10])

#loss
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
loss = []

with tf.Session() as sess:
    sess.run(init)
    x_test = mnist.test.images
    y_test_ = mnist.test.labels
    for i in range(3000):
        batch_x,batch_y = mnist.train.next_batch(100)
        #print(batch_y.shape)
        sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})
        sess.run(y,feed_dict={x:x_test})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_test_, 1))  # 1==line,0==column
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        loss.append(sess.run(cross_entropy,feed_dict={x:batch_x,y_:batch_y}))
        print(i)
        if i==999:
            print(sess.run(accuracy,feed_dict={x:x_test,y_:y_test_}))
    loss = np.array(loss)
    plt.plot(range(1000), loss)
    plt.show()