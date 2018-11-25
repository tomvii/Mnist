#测试该网路
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

###加载mnist_inference.py和mnist_train.py中定义的常量和前向传播的函数########  
import mnist_inference
import mnist_train
#每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率  
EVAL_INTERVAL_SECS = 10

def evaluate( mnist ):
    with tf.Graph().as_default() as g:#将默认图设为g 
        x = tf.placeholder(tf.float32, [mnist.validation.images.shape[0],
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS], name='x-input1')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        xs = mnist.validation.images
        # 类似地将输入的测试数据格式调整为一个四维矩阵 
        reshaped_xs = np.reshape(xs, (mnist.validation.images.shape[0],
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}
        #直接通过调用封装好的函数来计算前向传播的结果 
        #测试时不关注过拟合问题，所以正则化输入为None 
        print(reshaped_xs.shape)
        y = mnist_inference.inference(x,None,None)

    #使用前向传播的结果计算正确率，如果需要对未知的样例进行分类 
    #使用tf.argmax(y, 1)就可以得到输入样例的预测类别 
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 首先将一个布尔型的数组转换为实数，然后计算平均值 
    # 平均值就是网络在这一组数据上的正确率 
    #True为1，False为0 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #通过变量重命名的方式来加载模型 
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
    # 所有滑动平均的值组成的字典，处在/ExponentialMovingAverage下的值 
    # 为了方便加载时重命名滑动平均量，tf.train.ExponentialMovingAverage类 
    # 提供了variables_to_store函数来生成tf.train.Saver类所需要的变量 
        saver = tf.train.Saver(variable_to_restore) #这些值要从模型中提取 
    #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化 
    # #while True: 
        for i in range(2): # 为了降低个人电脑的压力，此处只利用最后生成的模型对测试数据集做测试 
            with tf.Session() as sess:
                #tf.train.get_checkpoint_state函数 
                # 会通过checkpoint文件自动找到目录中最新模型的文件名 
                ckpt = tf.train.get_checkpoint_state( mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型 
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #得到所有的滑动平均值 
                    #通过文件名得到模型保存时迭代的轮数 
                    global_step = ckpt.model_checkpoint_path.split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict = validate_feed) #使用此模型检验 
                    #没有初始化滑动平均值，只是调用模型的值，inference只是提供了一个变量的接口，完全没有赋值 
                    print("After %s training steps, validation accuracy = %g" %(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SECS)
        # time sleep()函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起的时间。 

def main( argv=None ):
    mnist = input_data.read_data_sets("./", one_hot=True)
    evaluate(mnist)

if __name__=='__main__':
    tf.app.run()
 
