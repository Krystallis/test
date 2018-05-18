import tensorflow as tf
import tensorlayer as tl

import layer

class Model:

    def __init__(self, sess):

        self.sess = sess
        self.neural_net()

    def neural_net(self):


        with tf.name_scope('input'):
            self.global_step = tf.Variable(0, trainable=False)
            self.drop_rate = tf.placeholder(tf.float32)
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 224, 224, 1], name='X')
            self.Y = tf.placeholder(tf.float32, [None, 224, 224, 2], name='Y')
            self.batch_size = tf.placeholder(tf.int32)

        with tf.name_scope('down'):
            conv1 = layer.conv2D('conv1_1', self.X, 32, [3, 3], [1, 1], 'same')
            conv1 = layer.BatchNorm('BN1-1', conv1, self.training)
            conv1 = layer.p_relu('act1_1', conv1)
            conv1 = layer.conv2D('conv1_2', conv1, 32, [3, 3], [1, 1], 'same')
            conv1 = layer.BatchNorm('BN1-2', conv1, self.training)
            conv1 = layer.p_relu('act1_2', conv1)
            # print(conv1.shape)
            pool1 = layer.maxpool('pool1', conv1, [2, 2], [2, 2], 'same') # 112 x 112
            # print(pool1.shape)

            conv2 = layer.conv2D('conv2_1', pool1, 64, [3, 3], [1, 1], 'same')
            conv2 = layer.BatchNorm('BN2-1', conv2, self.training)
            conv2 = layer.p_relu('act2_1', conv2)
            conv2 = layer.conv2D('conv2_2', conv2, 64, [3, 3], [1, 1], 'same')
            conv2 = layer.BatchNorm('BN2-2', conv2, self.training)
            conv2 = layer.p_relu('act2_2', conv2)
            # print(conv2.shape)
            pool2 = layer.maxpool('pool2', conv2, [2, 2], [2, 2], 'same') # 56 x 56
            # print(pool2.shape)

            conv3 = layer.conv2D('conv3_1', pool2, 128, [3, 3], [1, 1], 'same')
            conv3 = layer.BatchNorm('BN3-1', conv3, self.training)
            conv3 = layer.p_relu('act3_1', conv3)
            conv3 = layer.conv2D('conv3_2', conv3, 128, [3, 3], [1, 1], 'same')
            conv3 = layer.BatchNorm('BN3-2', conv3, self.training)
            conv3 = layer.p_relu('act3_2', conv3)
            # print(conv3.shape)
            pool3 = layer.maxpool('pool3', conv3, [2, 2], [2, 2], 'same') # 28 x 28
            # print(pool3.shape)

            conv4 = layer.conv2D('conv4_1', pool3, 256, [3, 3], [1, 1], 'same')
            conv4 = layer.BatchNorm('BN4-1', conv4, self.training)
            conv4 = layer.p_relu('act4_1', conv4)
            conv4 = layer.conv2D('conv4_2', conv4, 256, [3, 3], [1, 1], 'same')
            conv4 = layer.BatchNorm('BN4-2', conv4, self.training)
            conv4 = layer.p_relu('act4_2', conv4)
            # print(conv4.shape)
            pool4 = layer.maxpool('pool4', conv4, [2, 2], [2, 2], 'same') # 14 x 14
            # print(pool4.shape)

            conv5 = layer.conv2D('conv5_1', pool4, 512, [3, 3], [1, 1], 'same')
            conv5 = layer.BatchNorm('BN5-1', conv5, self.training)
            conv5 = layer.p_relu('act5_1', conv5)
            conv5 = layer.conv2D('conv5_2', conv5, 512, [3, 3], [1, 1], 'same')
            conv5 = layer.BatchNorm('BN5-2', conv5, self.training)
            conv5 = layer.p_relu('act5_2', conv5)
            # print(conv5.shape)


        with tf.name_scope('up'):
            up4 = layer.deconv2D('deconv4', conv5, [3, 3, 256, 512], [self.batch_size, 28, 28, 256], [1, 2, 2, 1], 'SAME')
            up4 = layer.BatchNorm('deBN4', up4, self.training)
            up4 = layer.p_relu('deact4', up4)
            # print(up4.shape)
            up4 = layer.concat('concat4', [up4, conv4], 3)
            # print(up4.shape)
            conv4 = layer.conv2D('uconv4_1', up4, 256, [3, 3], [1, 1], 'same')
            conv4 = layer.BatchNorm('uBN4-1', conv4, self.training)
            conv4 = layer.p_relu('uact4-1', conv4)
            conv4 = layer.conv2D('uconv4_2', conv4, 256, [3, 3], [1, 1], 'same')
            conv4 = layer.BatchNorm('uBN4-2', conv4, self.training)
            conv4 = layer.p_relu('uact4-2', conv4)
            # print(conv4.shape)

            up3 = layer.deconv2D('deconv3', conv4, [3, 3, 128, 256], [self.batch_size, 56, 56, 128], [1, 2, 2, 1], 'SAME')
            up3 = layer.BatchNorm('deBN3', up3, self.training)
            up3 = layer.p_relu('deact3', up3)
            # print(up3.shape)
            up3 = layer.concat('concat3', [up3, conv3], 3)
            # print(up3.shape)
            conv3 = layer.conv2D('uconv3_1', up3, 128, [3, 3], [1, 1], 'same')
            conv3 = layer.BatchNorm('uBN3-1', conv3, self.training)
            conv3 = layer.p_relu('uact3-1', conv3)
            conv3 = layer.conv2D('uconv3_2', conv3, 128, [3, 3], [1, 1], 'same')
            conv3 = layer.BatchNorm('uBN3-2', conv3, self.training)
            conv3 = layer.p_relu('uact3-2', conv3)
            # print(conv3.shape)

            up2 = layer.deconv2D('deconv2', conv3, [3, 3, 64, 128], [self.batch_size, 112, 112, 64], [1, 2, 2, 1], 'SAME')
            up2 = layer.BatchNorm('deBN2', up2, self.training)
            up2 = layer.p_relu('deact2', up2)
            # print(up2.shape)
            up2 = layer.concat('concat2', [up2, conv2], 3)
            # print(up2.shape)
            conv2 = layer.conv2D('uconv2_1', up2, 64, [3, 3], [1, 1], 'same')
            conv2 = layer.BatchNorm('uBN2-1', conv2, self.training)
            conv2 = layer.p_relu('uact2-1', conv2)
            conv2 = layer.conv2D('uconv2_2', conv2, 64, [3, 3], [1, 1], 'same')
            conv2 = layer.BatchNorm('uBN2-2', conv2, self.training)
            conv2 = layer.p_relu('uact2-2', conv2)
            # print(conv2.shape)

            up1 = layer.deconv2D('deconv1', conv2, [3, 3, 32, 64], [self.batch_size, 224, 224, 32], [1, 2, 2, 1], 'SAME')
            up1 = layer.BatchNorm('deBN1', up1, self.training)
            up1 = layer.p_relu('deact1', up1)
            # print(up1.shape)
            up1 = layer.concat('concat1', [up1, conv1], 3)
            # print(up1.shape)
            conv1 = layer.conv2D('uconv1_1', up1, 32, [3, 3], [1, 1], 'same')
            conv1 = layer.BatchNorm('uBN1-1', conv1, self.training)
            conv1 = layer.p_relu('uact1-1', conv1)
            conv1 = layer.conv2D('uconv1_2', conv1, 32, [3, 3], [1, 1], 'same')
            conv1 = layer.BatchNorm('uBN1-2', conv1, self.training)
            conv1 = layer.p_relu('uact1-2', conv1)

            out_seg = layer.conv2D('uconv1', conv1, 2, [1, 1], [1, 1], 'same')
            out_seg = layer.BatchNorm('out_BN', out_seg, self.training)
            out_seg = layer.p_relu('out_act',out_seg)
            # print(out_seg.shape)


        with tf.name_scope('optimizer'):

            # self.output = tl.act.pixel_wise_softmax(out_seg)

            # self.loss = 1 - tl.cost.dice_coe(self.output, self.Y)

            # self.loss = tl.cost.dice_hard_coe(self.output, self.Y)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_seg, labels=self.Y))

            #self.l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            #self.loss = tf.add(self.loss,self.l2_loss)

            self.init_learning = 0.01

            self.decay_step = 5000

            self.decay_rate = 0.9

            self.exponential_decay_learning_rate = tf.train.exponential_decay(
                learning_rate=self.init_learning,
                global_step=self.global_step,
                decay_steps=self.decay_step,
                decay_rate=self.decay_rate,
                staircase=True,
                name='learning_rate'
            )

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.exponential_decay_learning_rate, epsilon=0.00001)

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(self.update_ops):
                self.trainer = self.optimizer.minimize(self.loss, global_step=self.global_step)

            self.out = tf.nn.softmax(out_seg)

            self.predicted, _ = tf.split(self.out, [1, 1], 3)

            self.truth, _ = tf.split(self.Y, [1, 1], 3)

            self.accuracy = layer.iou_coe(output=self.predicted, target=self.truth)



    def train(self,x_data,y_data,batch_size):
        return self.sess.run([self.trainer, self.loss],
                             feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0.3, self.training: True,
                                        self.batch_size: batch_size})

    def get_accuracy(self,x_data,y_data,batch_size):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0, self.training: False,
                                        self.batch_size: batch_size})

    def show_result(self,test_image,batch_size):
        return self.sess.run(self.predicted, feed_dict={self.X: test_image, self.drop_rate: 0, self.training: False,
                                                        self.batch_size: batch_size})

    def learning_rate(self):
        return self.sess.run(self.exponential_decay_learning_rate)
