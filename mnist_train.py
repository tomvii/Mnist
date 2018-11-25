##lenet-5训练过程
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
 
import mnist_inference

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

##配置神经网络的参数
BATCH_SIZE=100
LEARNING_RATE_BASE=0.01  #基础学习率
LEARNING_RATE_DECAY=0.99
REGULARAZTION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99
##模型保存的路径和文件名
MODEL_SAVE_PATH="./model/"
MODEL_NAME="model.ckpt"
 
##定义训练过程
def train(mnist):
    x=tf.placeholder(tf.float32,[
    BATCH_SIZE,
    mnist_inference.IMAGE_SIZE,
    mnist_inference.IMAGE_SIZE,
    mnist_inference.NUM_CHANNELS],
                 name='x-input')
 
    
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE] , name='y-input')  
    regularizer=tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y=mnist_inference.inference(x,True,regularizer)
    global_step=tf.Variable(0,trainable=False)
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
 
    #在所有代表神经网络参数的变量上使用滑动平均。
    variables_averages_op=variable_averages.apply(tf.trainable_variables())
 
    #计算使用了滑动平均之后的前向传播结果。
    #average_y=inference(x,variable_average2,weights1,biases1,weights2,biases2)
 
    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    #cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))  
    #计算在当前batch中所有样例的交叉熵平均值
 
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
 
    #计算L2正则化损失函数
    #regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失
    #regularization=regularizer(weights1)+regularizer(weights2)
    #总损失等于交叉熵损失和正则化损失的和
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))                                            #regularization
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)  
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variables_averages_op]):train_op=tf.no_op(name='train')
    
    #初始化tensorflow持久化类
    saver=tf.train.Saver()
    ##初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("****************开始训练************************")  
       # validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
 
        #准备测试数据.
        #test_feed={x:mnist.test.images,y_:mnist.test.labels}
       
 
        #迭代地训练神经网络
        for  i in range(TRAINING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,  
                                          mnist_inference.IMAGE_SIZE,  
                                          mnist_inference.IMAGE_SIZE,  
                                          mnist_inference.NUM_CHANNELS))
            train_op_renew,loss_value, step=sess.run([train_op,loss,global_step],
                                       feed_dict={x:reshaped_xs,y_:ys})
 
            if i%1000==0:
                print("After %d training step(s),loss on training batch is %g."%(step,loss_value))
 
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
                           
def main(argv=None):
    mnist=input_data.read_data_sets("./",one_hot=True)
    train(mnist)
if __name__=='__main__':
    tf.app.run()
