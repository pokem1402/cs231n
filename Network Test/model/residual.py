import tensorflow as tf

def ver1(X,y,is_training):
    module_iter = 10
    num_classes = 10
    [N,W,H,C] = X.shape
    """
    setup variables
    """
    # 3x3 conv
    F0 = 64
    W3conv0 = tf.get_variable("W3conv0", shape=[3,3,C,F0])
    b3conv0 = tf.get_variable("b1conv0", shape=[F0])
    ##############################################################
    # Convolution layer 3x3 - [1x1]*FN
    ##############################################################
    W3conv1 = [tf.get_variable("W3conv1"+str(i), shape=[3,3,F0,F0]) for i in range(1,module_iter+1)]
    b3conv1 = [tf.get_variable("b3conv1"+str(i), shape=[F0]) for i in range(1,module_iter+1)]
    W1 = tf.get_variable("W1", shape= [32**2*F0, 1000])
    b1 = tf.get_variable("b1", shape=[1000])
    W2 = tf.get_variable("W2", shape= [1000, num_classes])
    b2= tf.get_variable("b2", shape=[num_classes])
    """
     define graph
    """
    out = tf.nn.conv2d(X,W3conv0,strides=[1,1,1,1], padding='SAME', name = "three")+b3conv0
    for i in range(int(module_iter/2)):
        out_t = tf.nn.conv2d(out, W3conv1[2*i], strides = [1,1,1,1], padding='SAME',name="three1")+b3conv1[2*i]
        out_t = tf.nn.relu(out_t)
        out_t = tf.nn.conv2d(out_t, W3conv1[2*i+1], strides = [1,1,1,1], padding='SAME',name="three1")+b3conv1[2*i+1]
        out = out_t + out
        out = tf.nn.relu(out)
    out = tf.nn.avg_pool(out, [1,3,3,1], [1,1,1,1], padding='SAME',name="avg1")
    out = tf.reshape(out, [-1,32*32*F0])
    out = tf.matmul(out, W1)+b1
    y_out = tf.matmul(out, W2)+b2
    return y_out

def ver2(X,y,is_training):
    module_iter = 10
    num_classes = 10
    [N,W,H,C] = X.shape
    """
    setup variables
    """
    # 3x3 conv
    F0 = 64
    W3conv0 = tf.get_variable("W3conv0", shape=[3,3,C,F0])
    b3conv0 = tf.get_variable("b1conv0", shape=[F0])
    ##############################################################
    # Convolution layer 3x3 - [1x1]*FN
    ##############################################################
    W3conv1 = [tf.get_variable("W3conv1"+str(i), shape=[3,3,F0,F0]) for i in range(1,module_iter+1)]
    b3conv1 = [tf.get_variable("b3conv1"+str(i), shape=[F0]) for i in range(1,module_iter+1)]
    W1conv1 = tf.get_variable("W1conv1", shape= [1,1,F0,1000])
    b1conv1 = tf.get_variable("b1conv1", shape=[1000])
    W1conv2 = tf.get_variable("W1conv2", shape= [1,1,1000, num_classes])
    b1conv2 = tf.get_variable("b1conv2", shape=[num_classes])
    """
     define graph
    """
    out = tf.nn.conv2d(X,W3conv0,strides=[1,1,1,1], padding='SAME', name = "three")+b3conv0
    for i in range(int(module_iter/2)):
        out_t = tf.nn.conv2d(out, W3conv1[2*i], strides = [1,1,1,1], padding='SAME',name="three1")+b3conv1[2*i]
        out_t = tf.nn.relu(out_t)
        out_t = tf.nn.conv2d(out_t, W3conv1[2*i+1], strides = [1,1,1,1], padding='SAME',name="three1")+b3conv1[2*i+1]
        out = out_t + out
        out = tf.nn.relu(out)
    out = tf.nn.conv2d(out, W1conv1, strides=[1,1,1,1], padding='VALID', name ='one_first') + b1conv1
    out = tf.nn.conv2d(out, W1conv2, strides=[1,1,1,1], padding='VALID', name ='one_last') + b1conv2
    out = tf.nn.avg_pool(out, [1,32,32,1], [1,1,1,1], padding='VALID', name='avg')
    y_out = tf.reshape(out, [-1, num_classes])
    return y_out
