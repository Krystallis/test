import tensorflow as tf
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

            self.init_learning = 0.01

            self.decay_step = 5000

            self.decay_rate = 0.9

            self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')
            self.Y = tf.placeholder(tf.int32, [None, 10], name='Y')


        with tf.name_scope('dense_layer1'):

            X1 = layer.conv2D('conv1', self.X, 32, [3, 3], [1, 1], 'same')
            X1 = layer.BatchNorm('bn1', X1, self.training)
            X1 = layer.p_relu('act1', X1)
            # print(X1.shape)

            X2 = layer.conv2D('conv2', X1, 32, [3, 3], [1, 1], 'same')
            X2 = layer.BatchNorm('bn2', X2, self.training)
            X2 = layer.p_relu('act2', X2)
            # print(X2.shape)

            X3 = layer.concat('concat1', [X1, X2])
            # print(X3.shape)
            X3 = layer.conv2D('conv3', X3, 32, [3, 3], [1, 1], 'same')
            X3 = layer.BatchNorm('bn3', X3, self.training)
            X3 = layer.p_relu('act3', X3)
            # print(X3.shape)

            X4 = layer.concat('concat2', [X1, X2, X3])
            # print(X4.shape)
            X4 = layer.conv2D('conv4', X4, 32, [3, 3], [1, 1], 'same')
            X4 = layer.BatchNorm('bn4', X4, self.training)
            X4 = layer.p_relu('act4', X4)
            # print(X4.shape)

            X5 = layer.concat('concat3', [X1, X2, X3, X4])
            # print(X5.shape)
            X5 = layer.conv2D('conv5', X5, 32, [3, 3], [1, 1], 'same')
            X5 = layer.BatchNorm('bn5', X5, self.training)
            X5 = layer.p_relu('act5', X5)
            # print(X5.shape)

        with tf.name_scope('middle_node1'):
            X5 = layer.conv2D('mconv1', X5, 64, [1, 1], [1, 1])
            # print(X5.shape)
            X5 = layer.maxpool('mp1', X5, [2, 2], [2, 2])
            # print(X5.shape)

        with tf.name_scope('dense_layer2'):
            X6 = layer.conv2D('conv6', X5, 64, [3, 3], [1, 1], 'same')
            X6 = layer.BatchNorm('bn6', X6, self.training)
            X6 = layer.p_relu('act6', X6)
            # print(X6.shape)

            X7 = layer.conv2D('conv7', X6, 64, [3, 3], [1, 1], 'same')
            X7 = layer.BatchNorm('bn7', X7, self.training)
            X7 = layer.p_relu('act7', X7)
            # print(X7.shape)

            X8 = layer.concat('concat4', [X6, X7])
            # print(X8.shape)
            X8 = layer.conv2D('conv8', X8, 64, [3, 3], [1, 1], 'same')
            X8 = layer.BatchNorm('bn8', X8, self.training)
            X8 = layer.p_relu('act8', X8)
            # print(X8.shape)

            X9 = layer.concat('concat5', [X6, X7, X8])
            # print(X9.shape)
            X9 = layer.conv2D('conv9', X9, 64, [3, 3], [1, 1], 'same')
            X9 = layer.BatchNorm('bn9', X9, self.training)
            X9 = layer.p_relu('act9', X9)
            # print(X9.shape)

            X10 = layer.concat('concat6', [X6, X7, X8, X9])
            # print(X10.shape)
            X10 = layer.conv2D('conv10', X10, 64, [3, 3], [1, 1], 'same')
            X10 = layer.BatchNorm('bn10', X10, self.training)
            X10 = layer.p_relu('act10', X10)
            # print(X10.shape)

        with tf.name_scope('middle_node2'):
            X10 = layer.conv2D('mconv2', X10, 128, [1, 1], [1, 1])
            # print(X10.shape)
            X10 = layer.maxpool('mp2', X10, [2, 2], [2, 2])
            # print(X10.shape)

        with tf.name_scope('dense_layer3'):
            X11 = layer.conv2D('conv11', X10, 128, [3, 3], [1, 1], 'same')
            X11 = layer.BatchNorm('bn11', X11, self.training)
            X11 = layer.p_relu('act11', X11)
            # print(X11.shape)

            X12 = layer.conv2D('conv12', X11, 128, [3, 3], [1, 1], 'same')
            X12 = layer.BatchNorm('bn12', X12, self.training)
            X12 = layer.p_relu('act12', X12)
            # print(X12.shape)

            X13 = layer.concat('concat7', [X11, X12])
            # print(X13.shape)
            X13 = layer.conv2D('conv13', X13, 128, [3, 3], [1, 1], 'same')
            X13 = layer.BatchNorm('bn13', X13, self.training)
            X13 = layer.p_relu('act13', X13)
            # print(X13.shape)

            X14 = layer.concat('concat8', [X11, X12, X13])
            # print(X14.shape)
            X14 = layer.conv2D('conv14', X14, 128, [3, 3], [1, 1], 'same')
            X14 = layer.BatchNorm('bn14', X14, self.training)
            X14 = layer.p_relu('act14', X14)
            # print(X14.shape)

            X15 = layer.concat('concat9', [X11, X12, X13, X14])
            # print(X15.shape)
            X15 = layer.conv2D('conv15', X15, 128, [3, 3], [1, 1], 'same')
            X15 = layer.BatchNorm('bn15', X15, self.training)
            X15 = layer.p_relu('act15', X15)
            # print(X15.shape)

        with tf.name_scope('middle_node3'):
            X15 = layer.conv2D('mconv3', X15, 256, [1, 1], [1, 1])
            # print(X15.shape)

        with tf.name_scope('dense_layer4'):
            X16 = layer.conv2D('conv16', X15, 256, [3, 3], [1, 1], 'same')
            X16 = layer.BatchNorm('bn16', X16, self.training)
            X16 = layer.p_relu('act16', X16)
            # print(X16.shape)

            X17 = layer.conv2D('conv17', X16, 256, [3, 3], [1, 1], 'same')
            X17 = layer.BatchNorm('bn17', X17, self.training)
            X17 = layer.p_relu('act17', X17)
            # print(X17.shape)

            X18 = layer.concat('concat10', [X16, X17])
            # print(X18.shape)
            X18 = layer.conv2D('conv18', X18, 256, [3, 3], [1, 1], 'same')
            X18 = layer.BatchNorm('bn18', X18, self.training)
            X18 = layer.p_relu('act18', X18)
            # print(X18.shape)

            X19 = layer.concat('concat11', [X16, X17, X18])
            # print(X19.shape)
            X19 = layer.conv2D('conv19', X19, 256, [3, 3], [1, 1], 'same')
            X19 = layer.BatchNorm('bn19', X19, self.training)
            X19 = layer.p_relu('act19', X19)
            # print(X19.shape)

            X20 = layer.concat('concat12', [X16, X17, X18, X19])
            # print(X20.shape)
            X20 = layer.conv2D('conv20', X20, 256, [3, 3], [1, 1], 'same')
            X20 = layer.BatchNorm('bn20', X20, self.training)
            X20 = layer.p_relu('act20', X20)
            # print(X20.shape)

        with tf.name_scope('GAP') as scope:
            X_gap = layer.s_conv2D('GAP_1', X20, 10, [1, 1], [1, 1], 'same')
            # print(X_gap.shape)
            X_gap = layer.averagepool('avp', X_gap, [8, 8], [1, 1])
            # print(X_gap.shape)
            self.logits = tf.squeeze(X_gap, [1, 2])
            # print(self.logits.shape)

        with tf.name_scope('optimizer'):

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

            # self.l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # self.loss = tf.add(self.loss, self.l2_loss)

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

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.exponential_decay_learning_rate, epsilon=0.0001)

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(self.update_ops):
                self.trainer = self.optimizer.minimize(self.loss, global_step=self.global_step, name='train')

            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1)), tf.float32))

            # tf.summary.scalar('loss', self.loss)
            # tf.summary.scalar('lr', self.exponential_decay_learning_rate)
            # tf.summary.scalar('accuracy', self.accuracy)

            # self.merged = tf.summary.merge_all()
            # self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

    def train(self,x_data,y_data):
        return self.sess.run([self.trainer, self.loss], feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0.3, self.training: True})

    def predict_result(self,x_data):
        return self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.X: x_data, self.drop_rate: 0, self.training: False})

    def get_accuracy(self,x_data,y_data):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0, self.training: False})

    # def tensorboard(self,x_data,y_data):
        # summary = self.sess.run(self.merged, feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0, self.training: False})
        # return self.writer.add_summary(summary, global_step=self.sess.run(self.global_step))

    def learning_rate(self):
        return self.sess.run(self.exponential_decay_learning_rate)



