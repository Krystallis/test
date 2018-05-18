import loader2
import tensorflow as tf
from densenet import Model
import time


test_image = 'E:/cifar10/test/'
test_label = 'E:/cifar10/test_label.csv'

testX = loader2.image_load(test_image)
testY = loader2.label_load(test_label)

tf.reset_default_graph()

with tf.Session() as sess:

    m = Model(sess)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    test_batch_size = 1000

    #saver.restore(sess, './model/fully_trained_cnn.ckpt')
    saver.restore(sess, './model/cnn.ckpt')
    print("LOAD MODEL")

    test_batch = int(len(testX) / test_batch_size)
    total_accuracy = 0
    print('TESTING START')

    for idx in range(test_batch):

        batch_test_xs, batch_test_ys = loader2.next_batch(testX, testY, idx, test_batch_size)

        batch_accuracy = m.get_accuracy(batch_test_xs, batch_test_ys)
        print(batch_accuracy)
        total_accuracy += batch_accuracy

    print('TOTAL_ACCURACY: %.2f%%' % (total_accuracy / test_batch * 100))