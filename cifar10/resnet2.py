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
            self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')

            self.Y = tf.placeholder(tf.int32, [None, 10], name='Y')


        with tf.name_scope('layer1'):
            self.layer = layer.conv2D('conv1-1',self.X, 30, [1, 5], [1, 1])
            self.layer = layer.BatchNorm('bn1',self.layer,self.training)
            self.layer = layer.p_relu('active1-1',self.layer)
            # print(self.layer.shape)
            self.layer = layer.conv2D('conv1-2',self.layer, 30, [5, 1], [1, 1])
            self.layer = layer.BatchNorm('bn2',self.layer, self.training)
            self.layer = layer.p_relu('active1-2',self.layer)
            # print(self.layer.shape)
            self.layer = layer.maxpool('mp1',self.layer, [2, 2], [2, 2])
            # print(self.layer.shape)

        with tf.name_scope('layer2'):
            self.layer = layer.conv2D('conv2-1',self.layer, 90, [1, 3], [1, 1])
            self.layer = layer.BatchNorm('bn3',self.layer, self.training)
            self.layer = layer.p_relu('active2-1',self.layer)
            # print(self.layer.shape)
            self.layer = layer.conv2D('conv2-2',self.layer, 90, [3, 1], [1, 1])
            self.layer = layer.BatchNorm('bn4',self.layer, self.training)
            self.layer = layer.p_relu('active2-2',self.layer)
            # print(self.layer.shape)
            self.layer = layer.maxpool('mp2',self.layer, [2, 2], [2, 2])
            # print(self.layer.shape)

        with tf.name_scope('layer3'):
            self.layer = layer.conv2D('conv3-1',self.layer, 270, [1, 2], [1, 1])
            self.layer = layer.BatchNorm('bn5',self.layer, self.training)
            self.layer = layer.p_relu('active3-1',self.layer)
            # print(self.layer.shape)
            self.layer = layer.conv2D('conv3-2',self.layer, 270, [2, 1], [1, 1])
            self.layer = layer.BatchNorm('bn6',self.layer, self.training)
            self.layer = layer.p_relu('active3-2',self.layer)
            # print(self.layer.shape)

        with tf.name_scope('middle_flow'):
            self.m_layer = self.layer
            for i in range(8):
                self.residual = self.m_layer
                self.m_layer = layer.s_conv2D('s_conv' + str(i) + '-1', self.m_layer, 540, [1, 1], [1, 1], 'same')
                self.m_layer = layer.BatchNorm('bn'+str(i)+'1',self.m_layer, self.training)
                self.m_layer = layer.p_relu('m_active' + str(i) + '-1', self.m_layer)

                self.m_layer = layer.s_conv2D('s_conv' + str(i) + '-2', self.m_layer, 540, [3, 3], [1, 1], 'same')
                self.m_layer = layer.BatchNorm('bn'+str(i)+'2',self.m_layer, self.training)
                self.m_layer = layer.p_relu('m_active' + str(i) + '-2', self.m_layer)

                self.m_layer = layer.dropout('m_dp', self.m_layer, self.drop_rate, self.training)

                self.m_layer = layer.s_conv2D('s_conv' + str(i) + '-3', self.m_layer, 540, [3, 3], [1, 1], 'same')
                self.m_layer = layer.BatchNorm('bn'+str(i)+'3',self.m_layer, self.training)
                self.m_layer = layer.p_relu('m_active' + str(i) + '-3', self.m_layer)
                # print(self.m_layer.shape)

                self.m_layer = layer.s_conv2D('s_conv' + str(i) + '-4', self.m_layer, 270, [1, 1], [1, 1], 'same')
                self.m_layer = layer.add(self.m_layer, self.residual, name='add' + str(i))
                self.m_layer = layer.p_relu('m_active' + str(i) + '-4', self.m_layer)
                # print(self.m_layer.shape)

        with tf.name_scope('Global_Average_Pooling'):
            self.layer = layer.s_conv2D('reduce_channel',self.m_layer, 10, [1, 1], [1, 1])
            # print(self.layer.shape)
            self.layer = layer.averagepool('avp',self.layer, [5, 5], [1, 1])
            # print(self.layer.shape)
            self.logits = tf.squeeze(self.layer, [1, 2],name='logits')
            # print(self.logits.shape)


        with tf.name_scope('optimizer'):

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

            # self.l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # self.loss = tf.add(self.loss, self.l2_loss)

            self.init_learning = 0.01

            self.decay_step = 2500

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

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('lr', self.exponential_decay_learning_rate)
            tf.summary.scalar('accuracy', self.accuracy)

            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

    def train(self,x_data,y_data):
        return self.sess.run([self.trainer, self.loss], feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0.3, self.training: True})

    def predict_result(self,x_data):
        return self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.X: x_data, self.drop_rate: 0, self.training: False})

    def get_accuracy(self,x_data,y_data):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0, self.training: False})

    def learning_rate(self):
        return self.sess.run(self.exponential_decay_learning_rate)

    def tensorboard(self,x_data,y_data):
        summary = self.sess.run(self.merged, feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0, self.training: False})
        return self.writer.add_summary(summary, global_step=self.sess.run(self.global_step))






