import tensorflow as tf

def ver1(X,y,is_training):
    module_iter = 10
    fully_layer_num = 3
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
    W1conv1 = [[tf.get_variable("W1conv1"+str(i)+"_"+str(f), shape=[1,1,F0,F0])  for f in range(fully_layer_num)] for i in range(1,module_iter+1)]
    b1conv1 = [[tf.get_variable("b1conv1"+str(i)+"_"+str(f), shape=[F0] )for f in range(fully_layer_num)] for i in range(1,module_iter+1)]
    ##############################################################
    # Convolution layer 1x1
    ##############################################################
    W1conv2 = tf.get_variable("W1conv2", shape=[1,1,F0,num_classes])
    b1conv2 = tf.get_variable("b1conv2", shape=[num_classes])
    """
     define graph
    """
    out = tf.nn.conv2d(X,W3conv0,strides=[1,1,1,1], padding='SAME', name = "three")+b3conv0
    for i in range(module_iter):
        out = tf.nn.conv2d(out, W3conv1[i], strides = [1,1,1,1], padding='SAME',name="three1")+b3conv1[i]
        for f in range(fully_layer_num):
            out = tf.nn.conv2d(out,W1conv1[i][f],strides=[1,1,1,1], padding='SAME', name = "one1")+b1conv1[i][f]
    out = tf.nn.conv2d(out,W1conv2, strides=[1,1,1,1], padding='VALID', name = "one_last")+b1conv2
    out = tf.nn.avg_pool(out, [1,32,32,1], [1,1,1,1], padding='VALID',name="avg1")
    y_out = tf.reshape(out, [-1, num_classes])
    return y_out

def ver2(X,y,is_training):
    #############################################################
    #   [1x1 conv - relu] x2 instead of 1x1 conv x3
    #############################################################
    module_iter = 10
    fully_layer_num = 2
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
    W1conv1 = [[tf.get_variable("W1conv1"+str(i)+"_"+str(f), shape=[1,1,F0,F0])  for f in range(fully_layer_num)] for i in range(1,module_iter+1)]
    b1conv1 = [[tf.get_variable("b1conv1"+str(i)+"_"+str(f), shape=[F0] )for f in range(fully_layer_num)] for i in range(1,module_iter+1)]
    ##############################################################
    # Convolution layer 1x1
    ##############################################################
    W1conv2 = tf.get_variable("W1conv2", shape=[1,1,F0,num_classes])
    b1conv2 = tf.get_variable("b1conv2", shape=[num_classes])
    """
     define graph
    """
    out = tf.nn.conv2d(X,W3conv0,strides=[1,1,1,1], padding='SAME', name = "three")+b3conv0
    for i in range(module_iter):
        out = tf.nn.conv2d(out, W3conv1[i], strides = [1,1,1,1], padding='SAME',name="three1")+b3conv1[i]
        for f in range(fully_layer_num):
            out = tf.nn.conv2d(out,W1conv1[i][f],strides=[1,1,1,1], padding='SAME', name = "one1")+b1conv1[i][f]
            out = tf.nn.relu(out)
    out = tf.nn.conv2d(out,W1conv2, strides=[1,1,1,1], padding='VALID', name = "one_last")+b1conv2
    out = tf.nn.avg_pool(out, [1,32,32,1], [1,1,1,1], padding='VALID',name="avg1")
    y_out = tf.reshape(out, [-1, num_classes])
    return y_out
