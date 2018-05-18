import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer()
regularizer = None # tf.contrib.layers.l2_regularizer(0.00001)
def conv2D(name,inputs, filters, kernel_size, strides, padding='valid'):
    conv2D = tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              use_bias=True,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              name=name)
    return conv2D

def s_conv2D(name,inputs,filters,kernel_size,strides,padding='valid'):
    s_conv2D = tf.layers.separable_conv2d(inputs=inputs,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding=padding,
                                          use_bias=True,
                                          depthwise_initializer=initializer,
                                          depthwise_regularizer=regularizer,
                                          pointwise_initializer=initializer,
                                          pointwise_regularizer=regularizer,
                                          name=name)
    return s_conv2D


def GlobalAveragePooling2D(input, n_class, name):
    """
    replace Fully Connected Layer.
    https://www.facebook.com/groups/smartbean/permalink/1708560322490187/
    https://github.com/AndersonJo/global-average-pooling/blob/master/global-average-pooling.ipynb
    :param input: a tensor of input
    :param n_class: a number of classification class
    :return: class
    """
    # gap_filter = resnet.create_variable('filter', shape=(1, 1, 128, 10))
    gap_filter = tf.get_variable(name='gap_filter', shape=[1, 1, input.get_shape()[-1], n_class], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
    layer = tf.nn.conv2d(input, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
    layer = tf.nn.avg_pool(layer, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
    layer = tf.reduce_mean(layer, axis=[1, 2])
    return layer


def relu(name,inputs):
    active_layer = tf.nn.relu(inputs,name=name)
    return active_layer

def leaky_relu(name,inputs,alpha=0.01):
    active_layer = tf.nn.leaky_relu(inputs,alpha,name=name)
    return active_layer

def elu(name,inputs):
    active_layer = tf.nn.elu(inputs,name=name)
    return active_layer

def p_relu(name,inputs):
    alphas = tf.get_variable(name, inputs.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def BatchNorm(name,inputs,training):
    BN_layer = tf.layers.batch_normalization(inputs,momentum=0.99,epsilon=1e-08,training=training,name=name)
    return BN_layer


def maxpool(name,inputs, pool_size, strides, padding='valid'):
    MP_layer = tf.layers.max_pooling2d(inputs, pool_size, strides, padding,name=name)
    return MP_layer


def averagepool(name,inputs, pool_size, strides, padding='valid'):
    AP_layer = tf.layers.average_pooling2d(inputs, pool_size, strides, padding,name=name)
    return AP_layer


def maxout(name,inputs, num_units):
    # num_units must multiple of axis
    MO_layer = tf.contrib.layers.maxout(inputs, num_units,name=name)
    return MO_layer

def concat(name,inputs,axis=3):
    con_layer = tf.concat(inputs,axis,name=name)
    return con_layer

def dropout(name,inputs,drop_rate,training):
    DP_layer = tf.layers.dropout(inputs,drop_rate,training=training,name=name)
    return DP_layer

def add(*inputs,name):
    layer = tf.add(*inputs,name=name)
    return layer

def flatten(name,inputs):
    L1 = tf.layers.flatten(inputs,name=name)
    return L1

def fc(name,inputs,units):
    L2 = tf.layers.dense(inputs,units,name=name,kernel_initializer=initializer,kernel_regularizer=regularizer)
    return L2
'''

def inception_A(inputs,filter,n,training):

    L1_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,filter,[1,n],1,'SAME',training)

    L1_3 = conv2D(L1_2,filter,[n,1],1,'SAME',training)


    L2_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L2_2 = conv2D(L2_1,filter,[n,n],1,'SAME',training)


    L3_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)


    L4_1 = averagepool(inputs,[2,2],1,'SAME')

    L4_2 = conv2D(L4_1,filter,[1,1],1,'SAME',training)

    out_layer = concat([L1_3,L2_2,L3_1,L4_2],3)

    return out_layer


def inception_B(inputs,filter,n,training):

    L1_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,filter,[1,n],1,'SAME',training)

    L1_3 = conv2D(L1_2,filter,[n,1],1,'SAME',training)

    L1_4 = conv2D(L1_3,filter,[1,n],1,'SAME',training)

    L1_5 = conv2D(L1_4,filter,[n,1],1,'SAME',training)


    L2_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L2_2 = conv2D(L2_1,filter,[1,n],1,'SAME',training)

    L2_3 = conv2D(L2_2,filter,[n,1],1,'SAME',training)


    L3_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)


    L4_1 = averagepool(inputs,[2,2],1,'SAME')

    L4_2 = conv2D(L4_1,filter,[1,1],1,'SAME',training)


    out_layer = concat([L1_5,L2_3,L3_1,L4_2],3)

    return out_layer


def inception_C(inputs,filter,n,training):

    L1_1 = conv2D(inputs, filter,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,filter,[1,n],1,'SAME',training)

    L1_3 = conv2D(L1_2,filter,[n,1],1,'SAME',training)

    L1_4_1 = conv2D(L1_3,filter,[n,1],1,'SAME',training)

    L1_4_2 = conv2D(L1_3,filter,[1,n],1,'SAME',training)


    L2_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)

    L2_2_1 = conv2D(L2_1,filter,[n,1],1,'SAME',training)

    L2_2_2 = conv2D(L2_1,filter,[1,n],1,'SAME',training)


    L3_1 = conv2D(inputs,filter,[1,1],1,'SAME',training)


    L4_1 = averagepool(inputs,[2,2],1,'SAME')

    L4_2 = conv2D(L4_1,filter,[1,1],1,'SAME',training)


    out_layer = concat([L1_4_1,L1_4_2,L2_2_1,L2_2_2,L3_1,L4_2],3)

    return out_layer


def Reduction_A(inputs,m,n,k,l,training):

    L1_1 = conv2D(inputs,k,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,l,[3,3],1,'SAME',training)

    L1_3 = conv2D(L1_2,m,[3,3],2,'VALID',training)

    L2_1 = conv2D(inputs,n,[3,3],2,'VALID',training)

    L3_1 = maxpool(inputs,[3,3],2,'VALID')

    out_layer = concat([L1_3,L2_1,L3_1],3)

    return out_layer


def Reduction_B(inputs,training):

    L1_1 = conv2D(inputs,256,[1,1],1,'SAME',training)

    L1_2 = conv2D(L1_1,256,[1,7],1,'SAME',training)

    L1_3 = conv2D(L1_2,320,[7,1],1,'SAME',training)

    L1_4 = conv2D(L1_3,320,[3,3],2,'VALID',training)


    L2_1 = conv2D(inputs,192,[1,1],1,'SAME',training)

    L2_2 = conv2D(L2_1,192,[3,3],2,'VALID',training)


    L3_1 = maxpool(inputs, [3,3],2,'VALID')

    out_layer = concat([L1_4,L2_2,L3_1],3)

    return out_layer


def auxiliary_classifiers(inputs, filter, drop_rate, training):

    L1 = conv2D(inputs,filter,[3,3],1,'SAME',training)

    L2 = maxpool(L1,[2,2],2,'SAME')

    L3 = conv2D(L2,filter*1.2,[3,3],1,'SAME',training)

    L4 = maxpool(L3,[2,2],2,'SAME')

    flat = flatten(L4)

    L5 = fc(flat,650)

    L6 = dropout(L5,drop_rate,training)

    L7 = fc(L6,2)

    return L7

'''