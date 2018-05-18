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


        with tf.name_scope('layer1') as scope:
            X = layer.conv2D('conv1-1', self.X, 30, [7, 7], [1, 1], 'same')  # 32 x 32
            X = layer.BatchNorm('BN1-1', X, self.training)
            X = layer.p_relu('active1-1', X)
            print(X.shape)
            X = layer.maxpool('mp1', X, [2, 2], [2, 2])  # 16 x 16
            print(X.shape)

        with tf.name_scope('layer2') as scope:
            X = layer.conv2D('conv2-1', X, 60, [3, 3], [1, 1], 'same')  # 16 x 16
            X = layer.BatchNorm('BN2-1', X, self.training)
            X = layer.p_relu('active2-1', X)
            print(X.shape)
            X = layer.maxpool('mp2', X, [2, 2], [2, 2])  # 8 x 8
            print(X.shape)

        with tf.name_scope('layer3') as scope:
            X = layer.conv2D('conv3-1', X, 120, [3, 3], [1, 1], 'same')  # 8 x 8
            X = layer.BatchNorm('BN3-1', X, self.training)
            X = layer.p_relu('active3-1', X)
            print(X.shape)
            # X = layer.maxpool('mp3', X, [2, 2], [2, 2])  # 4 x 4

        with tf.name_scope('bottleneck1') as scope:
            RX = layer.conv2D('rconv1-1', X, 120, [1, 1], [1, 1], 'same')
            RX = layer.BatchNorm('rBN1-1', RX, self.training)

            X = layer.conv2D('bconv1-1', X, 60, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN1-1', X, self.training)
            X = layer.p_relu('bactive1-1', X)
            X = layer.conv2D('bconv1-2', X, 60, [3, 3], [1, 1], 'same')
            X = layer.BatchNorm('bBN1-2', X, self.training)
            X = layer.p_relu('bactive1-2', X)
            # X = layer.dropout('dp1', X, self.drop_rate, self.training)
            X = layer.conv2D('bconv1-3', X, 120, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN1-3', X, self.training)
            X = layer.add(X, RX, name='add1')
            X = layer.p_relu('bactive1-3', X)
            print(X.shape)

        with tf.name_scope('bottleneck2') as scope:
            RX = layer.conv2D('rconv2-1', X, 240, [1, 1], [1, 1], 'same')
            RX = layer.BatchNorm('rBN2-1', RX, self.training)

            X = layer.conv2D('bconv2-1', X, 120, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN2-1', X, self.training)
            X = layer.p_relu('bactive2-1', X)
            X = layer.conv2D('bconv2-2', X, 120, [3, 3], [1, 1], 'same')
            X = layer.BatchNorm('bBN2-2', X, self.training)
            X = layer.p_relu('bactive2-2', X)
            # X = layer.dropout('dp2', X, self.drop_rate, self.training)
            X = layer.conv2D('bconv2-3', X, 240, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN2-3', X, self.training)
            X = layer.add(X, RX, name='add2')
            X = layer.p_relu('bactive2-3', X)
            print(X.shape)

        with tf.name_scope('bottleneck3') as scope:
            RX = layer.conv2D('rconv3-1', X, 360, [1, 1], [1, 1], 'same')
            RX = layer.BatchNorm('rBN3-1', RX, self.training)

            X = layer.conv2D('bconv3-1', X, 240, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN3-1', X, self.training)
            X = layer.p_relu('bactive3-1', X)
            X = layer.conv2D('bconv3-2', X, 240, [3, 3], [1, 1], 'same')
            X = layer.BatchNorm('bBN3-2', X, self.training)
            X = layer.p_relu('bactive3-2', X)
            # X = layer.dropout('dp3', X, self.drop_rate, self.training)
            X = layer.conv2D('bconv3-3', X, 360, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN3-3', X, self.training)
            X = layer.add(X, RX, name='add3')
            X = layer.p_relu('bactive3-3', X)
            print(X.shape)

        with tf.name_scope('bottleneck4') as scope:
            RX = layer.conv2D('rconv4-1', X, 480, [1, 1], [1, 1], 'same')
            RX = layer.BatchNorm('rBN4-1', RX, self.training)

            X = layer.conv2D('bconv4-1', X, 360, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN4-1', X, self.training)
            X = layer.p_relu('bactive4-1', X)
            X = layer.conv2D('bconv4-2', X, 360, [3, 3], [1, 1], 'same')
            X = layer.BatchNorm('bBN4-2', X, self.training)
            X = layer.p_relu('bactive4-2', X)
            # X = layer.dropout('dp4', X, self.drop_rate, self.training)
            X = layer.conv2D('bconv4-3', X, 480, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN4-3', X, self.training)
            X = layer.add(X, RX, name='add4')
            X = layer.p_relu('bactive4-3', X)
            print(X.shape)
        '''
        with tf.name_scope('bottleneck5') as scope:
            RX = layer.conv2D('rconv5-1', X, 720, [1, 1], [1, 1], 'same')
            RX = layer.BatchNorm('rBN5-1', RX, self.training)

            X = layer.conv2D('bconv5-1', X, 480, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN5-1', X, self.training)
            X = layer.p_relu('bactive5-1', X)
            X = layer.conv2D('bconv5-2', X, 480, [3, 3], [1, 1], 'same')
            X = layer.BatchNorm('bBN5-2', X, self.training)
            X = layer.p_relu('bactive5-2', X)
            X = layer.dropout('dp5', X, self.drop_rate, self.training)
            X = layer.conv2D('bconv5-3', X, 720, [1, 1], [1, 1], 'same')
            X = layer.BatchNorm('bBN5-3', X, self.training)
            X = layer.add(X, RX, name='add5')
            X = layer.p_relu('bactive5-3', X)
            print(X.shape)
        '''

        with tf.name_scope('GAP') as scope:
            X = layer.conv2D('GAP_1', X, 10, [1, 1], [1, 1], 'same')
            print(X.shape)
            X = layer.averagepool('avp', X, [8, 8], [1, 1])
            print(X.shape)
            self.logits = tf.squeeze(X, [1, 2])
            print(self.logits.shape)

        with tf.name_scope('optimizer'):

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

            self.l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.loss = tf.add(self.loss, self.l2_loss)

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

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

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

    def tensorboard(self,x_data,y_data):
        summary = self.sess.run(self.merged, feed_dict={self.X: x_data, self.Y: y_data, self.drop_rate: 0, self.training: False})
        return self.writer.add_summary(summary, global_step=self.sess.run(self.global_step))

    def learning_rate(self):
        return self.sess.run(self.exponential_decay_learning_rate)



