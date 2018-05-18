import loader2
import tensorflow as tf
from densenet import Model
import time
import csv


train_image = 'E:/tensorflowdata/cifar10/train/'
train_label = 'E:/tensorflowdata/cifar10/train_label.csv'

test_image = 'E:/tensorflowdata/cifar10/test/'
test_label = 'E:/tensorflowdata/cifar10/test_label.csv'

print("LOADING DATA")
start = time.time()

trainX = loader2.image_load(train_image)
trainY = loader2.label_load(train_label)

testX = loader2.image_load(test_image)
testY = loader2.label_load(test_label)

end = time.time()

print(trainX.shape, trainY.shape, testX.shape, testY.shape)
print("DATA LOAD COMPLETE, ESTIMATE TIME: %d" % (int(end)-int(start)))
print("TRAIN:",len(trainX),'TEST:',len(testX))

with tf.Session() as sess:

    m = Model(sess)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    cost_list = []
    train_acc_list = []
    vali_acc_list = []
    lr_list = []

    print("BEGIN TRAINING")

    epoch_num = 150
    total_training_time = 0
    for epoch in range(epoch_num):
        start = time.time()

        total_cost = 0
        batch_size = 100
        train_step = int(len(trainX) / batch_size)

        for idx in range(train_step):

            batch_xs,batch_ys = loader2.next_batch(trainX,trainY,idx,batch_size)
            _, cost = m.train(batch_xs, batch_ys)
            total_cost += cost
            # m.tensorboard(batch_xs,batch_ys)

        test_batch_size = 1000
        train_accuracy = 0
        vali_accuracy = 0
        vali_step = int(len(testX) / test_batch_size)

        for idx in range(vali_step):

            batch_xs,batch_ys = loader2.next_batch(trainX,trainY,idx,test_batch_size)
            t_ac = m.get_accuracy(batch_xs, batch_ys)
            train_accuracy += t_ac

        for idx in range(vali_step):

            valiX,valiY = loader2.next_batch(testX,testY,idx,test_batch_size)

            vali_acc = m.get_accuracy(valiX, valiY)
            vali_accuracy += vali_acc

        end = time.time()
        training_time = end - start
        total_training_time += training_time
        lr = round(m.learning_rate(), 5)
        print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % epoch_num,
              'Loss =', '{:.8f}  '.format(total_cost / train_step),
              'Training time: {:.2f}  '.format(training_time),
              "Training Accuracy: %.2f%%  " % float(train_accuracy / vali_step * 100),
              "Validation Accuracy: %.2f%%  " % float(vali_accuracy / vali_step * 100),
              "Current Learning Rate:", lr)

        cost_list.append(total_cost / train_step)
        train_acc_list.append(train_accuracy / vali_step * 100)
        vali_acc_list.append(vali_accuracy / vali_step * 100)
        lr_list.append(lr)


        #if (epoch + 1) % 50 == 0:
        #    saver.save(sess, './model/cnn.ckpt')
        #    print("Model Saved")

    print("TRAINING COMPLETE")

    saver.save(sess, './model/fully_trained_cnn.ckpt')

    print("MODEL SAVED AT ./MODEL")

    print("TOTAL TRAINING TIME: %.3f" % total_training_time)

    file = open('./resnet_result.csv', 'w')
    writer = csv.writer(file, delimiter=',')
    for i in range(epoch_num):
        data = [i + 1, cost_list[i], train_acc_list[i], vali_acc_list[i], lr_list[i]]
        writer.writerow(data)
    file.close()

tf.reset_default_graph()

with tf.Session() as sess:

    m = Model(sess)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    saver.restore(sess, './model/fully_trained_cnn.ckpt')
    print("LOAD MODEL")

    test_batch_size = 1000

    test_batch = int(len(testX) / test_batch_size)
    total_accuracy = 0
    print('TESTING START')

    for idx in range(test_batch):

        batch_test_xs, batch_test_ys = loader2.next_batch(testX, testY, idx, test_batch_size)

        batch_accuracy = m.get_accuracy(batch_test_xs, batch_test_ys)
        total_accuracy += batch_accuracy

    print('TOTAL_ACCURACY: %.2f%%' % (total_accuracy * 100 / test_batch))