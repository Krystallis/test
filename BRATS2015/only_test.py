import loader
import tensorflow as tf
from unet import Model
import cv2
import numpy as np
import scipy.misc as sm

with tf.Session() as sess:

    m = Model(sess)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    saver.restore(sess, './model/fully_trained_cnn.ckpt')

    print("LOAD MODEL")

    test_image_path = 'E:\\tensorflowdata\\BRATS_ORI\\training\\'

    result_save_path = 'E:\\tensorflowdata\\BRATS_RESULT\\'

    data, _ = loader.data_list_load(test_image_path, "Flair")  # image_type = 'Flair' or 'T1' or 'T1c' or 'T2'

    for idx,image in enumerate(data):

        image = loader.read_image_grey_resized(image)

        predicted_result = m.show_result(image, 1)

        # 나올때의 shape = [1, 224, 224, 1]
        G = np.zeros([1,224,224,1])
        B = np.zeros([1,224,224,1])
        R = predicted_result

        predicted_result = np.concatenate((R,G,B),axis=3)
        predicted_result = np.squeeze(predicted_result)
        predicted_result = cv2.resize(predicted_result, (240, 240), interpolation=cv2.INTER_AREA)

        tR = image
        tG = image
        tB = image
        test_image = np.concatenate((tR,tG,tB),axis=3)
        test_image = np.squeeze(test_image)
        test_image = cv2.resize(test_image, (240, 240), interpolation=cv2.INTER_AREA)

        test_image = test_image.astype(float)

        # predicted_result = predicted_result*255
        w = 38

        dst = cv2.addWeighted(predicted_result, float(100 - w) * 0.0001, test_image, float(w) * 0.0001, 0)
        filename = result_save_path + str(idx) + '.jpg'
        sm.imsave(filename, dst)
        # cv2.imwrite(filename, dst)
        print(idx)

