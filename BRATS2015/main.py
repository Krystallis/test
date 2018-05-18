import tensorflow as tf
import loader
import unet
import time

training_path = 'E:\\tensorflowdata\\BRATS_ORI\\training\\'

data, label = loader.data_list_load(training_path,"Flair") # image_type = 'Flair' or 'T1' or 'T1c' or 'T2'

print('DATA:',len(data),'LABEL:', len(label))

data, label = loader.data_shuffle(data, label)

trainX, valiX, trainY, valiY = loader.TT_split(data,label)


print('train:',len(trainX),'vali:',len(valiX))



with tf.Session() as sess:
    m = unet.Model(sess)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    print("BEGIN TRAINING")

    epoch_num = 100
    total_training_time = 0
    for epoch in range(epoch_num):

        start = time.time()

        total_cost = 0
        train_batch_size = 10
        vali_batch_size = 100
        train_step = int(len(trainX) / train_batch_size)
        vali_step = int(len(valiX) / vali_batch_size)
        training_acc = 0
        total_vali_acc = 0
        step = 0

        trainX, trainY = loader.data_shuffle(trainX, trainY)

        for idx in range(train_step):
            batch_xs_list, batch_ys_list = loader.next_batch(trainX, trainY, idx, train_batch_size)
            batch_xs = loader.read_image_grey_resized(batch_xs_list)
            batch_ys = loader.read_label_grey_resized(batch_ys_list)
            _, cost = m.train(batch_xs, batch_ys,train_batch_size)
            total_cost += cost
            step += 1
            # print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % epoch_num,'step:',step,'  mini batch loss:', cost)

        for idx in range(train_step):
            training_batch_xs_list, training_batch_ys_list = loader.next_batch(trainX, trainY, idx, train_batch_size)
            training_batch_xs = loader.read_image_grey_resized(training_batch_xs_list)
            training_batch_ys = loader.read_label_grey_resized(training_batch_ys_list)
            acc = m.get_accuracy(training_batch_xs, training_batch_ys,train_batch_size)
            training_acc += acc

        for idx in range(vali_step):
            vali_batch_xs_list, vali_batch_ys_list = loader.next_batch(valiX, valiY, idx, vali_batch_size)
            vali_batch_xs = loader.read_image_grey_resized(vali_batch_xs_list)
            vali_batch_ys = loader.read_label_grey_resized(vali_batch_ys_list)
            vali_acc = m.get_accuracy(vali_batch_xs, vali_batch_ys,vali_batch_size)
            total_vali_acc += vali_acc


        end = time.time()
        training_time = end - start
        total_training_time += training_time

        print('Epoch:', '[%d' % (epoch + 1), '/ %d]   ' % epoch_num,
              'Loss =', '{:.10f}   '.format(total_cost / train_step),
              'Training Accuracy:{:.4f}   ' .format(training_acc / train_step),
              'Validation Accuracy:{:.4f}   '.format(total_vali_acc / vali_step),
              'Training time: {:.2f}   '.format(training_time),
              # 'Learning_rate:',m.learning_rate()
              )


    print("TRAINING COMPLETE")

    saver.save(sess, './model/fully_trained_cnn.ckpt')

    print("MODEL SAVED AT ./MODEL")

    print("TOTAL TRAINING TIME: %.3f" % total_training_time)


