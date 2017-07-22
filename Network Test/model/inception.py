import tensorflow as tf
import numpy as np
def abc():
    print("abc")

def ver1(X,y,is_training):
    """

    Hyper-parameters

    """
    num_classes = 10 # dedicated CIFAR-10
    module_iter = 10
    affine_output = 1024
    MO = 64
    [N,W,H,C] = X.shape
    """
    setup variables
    """
    # 1x1 conv
    W1conv0 = tf.get_variable("W1conv0", shape=[1,1,C,MO])
    b1conv0 = tf.get_variable("b1conv0",shape=[MO])
    ##############################################################
    #
    # inception module v1
    #
    # layer | 1x1 conv + 3x3 conv + 5x5 conv + [1x1 conv - max pool]
    #   C    |      4             4               4               4         = 16
    ##############################################################
    # 1x1 conv
    W1conv = [tf.get_variable("W1conv"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv = [tf.get_variable("b1conv"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    # 3x3 conv
    W3conv = [tf.get_variable("W3conv"+str(i), shape=[3,3,MO,MO/4]) for i in range(1,module_iter+1)]
    b3conv = [tf.get_variable("b3conv"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # 5x5 conv
    W5conv = [tf.get_variable("W5conv"+str(i), shape=[5,5,MO,MO/4]) for i in range(1,module_iter+1)]
    b5conv = [tf.get_variable("b5conv"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # for max_pool
    W1conv1 = [tf.get_variable("W1conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv1 = [tf.get_variable("b1conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    ##############################################################
    #
    #  Fully connected (affine x2)
    #
    ##############################################################
   # affine 1
    W1 = tf.get_variable("W1", shape=[MO/4*int(W*H), affine_output])
    b1 = tf.get_variable("b1", shape=[affine_output]) # F
    # affine 2
    W2 = tf.get_variable("W2", shape=[affine_output,num_classes])
    b2= tf.get_variable("b2", shape=[num_classes]) # F
    """
     define graph
    """
    out = X
    ##############################################################
    #
    #  1. max pool (2x2) stride = 2
    #
    #  To reduce window size
    #
    #  INPUT : [N,W,H,C]
    # OUTPUT : [N,W/2,H/2,C]
    ##############################################################
    out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], padding='VALID')
    out = tf.nn.conv2d(out, W1conv0, strides=[1,1,1,1], padding='VALID', name='init')+b1conv0
    ##############################################################
    #
    #  2. inception_moduel
    #
    #  Concatenate result of 1x1 conv, 3x3conv, 5x5 conv, max pool
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [N,W/2, H/2, C]
    #
    ##############################################################
    for i in range(module_iter):

        out1 = tf.nn.conv2d(out, W1conv[i], strides=[1,1,1,1], padding='VALID', name="one")+b1conv[i]
        out3 = tf.nn.conv2d(out, W3conv[i], strides = [1,1,1,1], padding='SAME',name="three")+b3conv[i]
        out5 = tf.nn.conv2d(out, W5conv[i], strides = [1,1,1,1], padding='SAME',name="five")+b5conv[i]
        out_max = tf.nn.conv2d(out, W1conv1[i], strides=[1,1,1,1], padding='SAME',name="one1")+b1conv1[i]
        out_max = tf.nn.max_pool(out_max, [1,3,3,1], [1,1,1,1], padding='SAME',name="max")
        out = tf.concat([out1, out3, out5, out_max], 3)
        out = tf.nn.relu(out)
    ##############################################################
    #
    #  3. Fully Connected
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [#Class]
    ##############################################################
    out = tf.reshape(out,[-1, int(MO/4)*int(W*H)])
    out = tf.matmul(out,W1)+b1
    y_out = tf.matmul(out,W2)+b2
    return y_out

def ver2(X,y,is_training):

    """
    Hyper-parameters
    """
    module_iter = 10
    affine_output = 1024
    num_classes = 10
    MO = 64
    [N,W,H,C] = X.shape
    """
    setup variables
    """
    # 1x1 conv
    W1conv0 = tf.get_variable("W1conv0", shape=[1,1,C,MO])
    b1conv0 = tf.get_variable("b1conv0",shape=[MO])
    ##############################################################
    #
    # inception module v2
    #
    # layer | 1x1 conv + (3x3 conv) + [3x3 conv - 3x3 conv] + [1x1 conv - max pool]
    #   C   |      4          4                 4               4         = 16
    ##############################################################
    # 1x1 conv
    W1conv = [tf.get_variable("W1conv"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv = [tf.get_variable("b1conv"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    # 3x3 conv
    W3conv = [tf.get_variable("W3conv"+str(i), shape=[3,3,MO,MO/4]) for i in range(1,module_iter+1)]
    b3conv = [tf.get_variable("b3conv"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # 3x3 conv
    W5conv1 = [tf.get_variable("W5conv1"+str(i), shape=[3,3,MO,MO/4]) for i in range(1,module_iter+1)]
    b5conv1 = [tf.get_variable("b5conv1"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    W5conv2 = [tf.get_variable("W5conv2"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b5conv2 = [tf.get_variable("b5conv2"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # for max pool1
    W1conv1 = [tf.get_variable("W1conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv1 = [tf.get_variable("b1conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    ##############################################################
    #
    #  Fully connected (affine x2)
    #
    ##############################################################
   # affine 1
    W1 = tf.get_variable("W1", shape=[MO/4*int(W*H), affine_output])
    b1 = tf.get_variable("b1", shape=[affine_output]) # F
    # affine 2
    W2 = tf.get_variable("W2", shape=[affine_output,num_classes])
    b2= tf.get_variable("b2", shape=[num_classes]) # F

    """
     define graph
    """
    out = X
    ##############################################################
    #
    #  1. max pool (2x2) stride = 2
    #
    #  To reduce window size
    #
    #  INPUT : [N,W,H,C]
    # OUTPUT : [N,W/2,H/2,C]
    ##############################################################

    out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], padding='VALID')
    out = tf.nn.conv2d(out, W1conv0, strides=[1,1,1,1], padding='VALID', name='init')+b1conv0
    ##############################################################
    #
    #  2. inception_moduel
    #
    #  Concatenate result of 1x1 conv, 3x3conv, 5x5 conv, max pool
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [N,W/2, H/2, C]
    #
    ##############################################################
    for i in range(module_iter):
        out1 = tf.nn.conv2d(out, W1conv[i], strides=[1,1,1,1], padding='VALID', name="one")+b1conv[i]
        out3 = tf.nn.conv2d(out, W3conv[i], strides = [1,1,1,1], padding='SAME',name="three")+b3conv[i]
        out5 = tf.nn.conv2d(out, W5conv1[i], strides = [1,1,1,1], padding='SAME',name="five1")+b5conv1[i]
        out5 = tf.nn.conv2d(out5, W5conv2[i], strides = [1,1,1,1], padding='SAME',name="five2")+b5conv2[i]
        out_max = tf.nn.conv2d(out, W1conv1[i], strides=[1,1,1,1], padding='SAME',name="one1")+b1conv1[i]
        out_max = tf.nn.max_pool(out_max, [1,3,3,1], [1,1,1,1], padding='SAME',name="max")
        out = tf.concat([out1, out3, out5, out_max], 3)
        out = tf.nn.relu(out)
    ##############################################################
    #
    #  3. Fully Connected
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [#Class]
    ##############################################################
    out = tf.reshape(out,[-1, int(MO/4)*int(W*H)])
    out = tf.matmul(out,W1)+b1
    y_out = tf.matmul(out,W2)+b2
    return y_out

def ver3(X,y,is_training):
    module_iter = 10
    affine_output = 1024
    num_classes = 10
    MO = 64 # module output/4
    [N,W,H,C] = X.shape
    """
    setup variables
    """
    # 1x1 conv
    W1conv0 = tf.get_variable("W1conv0", shape=[1,1,C,MO])
    b1conv0 = tf.get_variable("b1conv0",shape=[MO])
    ##############################################################
    #
    # inception module v3
    #
    # layer | 1x1 conv + (1x1conv-3x3 conv) + [1x1-conv-3x3 conv - 3x3 conv] + [max pool-1x1 conv]
    #   C    |      4             4               4               4         = 16
    ##############################################################
    # 1x1 conv
    W1conv1 = [tf.get_variable("W1conv"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv1 = [tf.get_variable("b1conv"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    # 3x3 conv
    W3conv1= [tf.get_variable("W3conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b3conv1 = [tf.get_variable("b3conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    W3conv2 = [tf.get_variable("W3conv2"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b3conv2 = [tf.get_variable("b3conv2"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # 5x5 conv
    W5conv1= [tf.get_variable("W5conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b5conv1 = [tf.get_variable("b5conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    W5conv2 = [tf.get_variable("W5conv2"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b5conv2 = [tf.get_variable("b5conv2"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    W5conv3 = [tf.get_variable("W5conv3"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b5conv3 = [tf.get_variable("b5conv3"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # max pool
    W1conv2 = [tf.get_variable("W1conv2"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv2 = [tf.get_variable("b1conv2"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    ##############################################################
    #
    #  Fully connected (affine x2)
    #
    ##############################################################
   # affine 1
    W1 = tf.get_variable("W1", shape=[int(MO/4)*int(W*H), affine_output])
    b1 = tf.get_variable("b1", shape=[affine_output]) # F
    # affine 2
    W2 = tf.get_variable("W2", shape=[affine_output,num_classes])
    b2= tf.get_variable("b2", shape=[num_classes]) # F

    """
     define graph
    """
    out = X
    ##############################################################
    #
    #  1. max pool (2x2) stride = 2
    #
    #  To reduce window size
    #
    #  INPUT : [N,W,H,C]
    # OUTPUT : [N,W/2,H/2,C]
    ##############################################################
    out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], padding='VALID')
    out = tf.nn.conv2d(out, W1conv0, strides=[1,1,1,1], padding='VALID', name='init')+b1conv0
    ##############################################################
    #
    #  2. inception_moduel
    #
    #  Concatenate result of 1x1 conv, 3x3conv, 5x5 conv, max pool
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [N,W/2, H/2, C]
    #
    ##############################################################
    for i in range(module_iter):
        # 1x1 conv
        out1 = tf.nn.conv2d(out, W1conv1[i], strides=[1,1,1,1], padding='VALID', name="one")+b1conv1[i]
        # 3x3 conv
        out3 = tf.nn.conv2d(out,W3conv1[i],strides=[1,1,1,1], padding='SAME', name = "three1")+b3conv1[i]
        out3 = tf.nn.conv2d(out3, W3conv2[i], strides = [1,1,1,1], padding='SAME',name="three2")+b3conv2[i]
        # 5x5 conv
        out5 = tf.nn.conv2d(out, W5conv1[i], strides=[1,1,1,1], padding='SAME', name="five1")+b5conv1[i]
        out5 = tf.nn.conv2d(out5, W5conv2[i], strides = [1,1,1,1], padding='SAME',name="five2")+b5conv2[i]
        out5 = tf.nn.conv2d(out5, W5conv3[i], strides = [1,1,1,1], padding='SAME',name="five3")+b5conv3[i]
        # max
        out_max = tf.nn.max_pool(out, [1,3,3,1], [1,1,1,1], padding='SAME',name="max1")
        out_max = tf.nn.conv2d(out_max, W1conv2[i], strides=[1,1,1,1], padding='SAME', name="max2")+b1conv2[i]
        out = tf.concat([out1, out3, out5, out_max], 3)
        out = tf.nn.relu(out)
    ##############################################################
    #
    #  3. Fully Connected
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [#Class]
    ##############################################################
    out = tf.reshape(out,[-1,int(MO/4)*int(W*H)])
    out = tf.matmul(out,W1)+b1
    y_out = tf.matmul(out,W2)+b2
    return y_out

def ver3_batch(X,y,is_training):
    module_iter = 10
    affine_output = 1024
    num_classes = 10
    MO = 64 # module output/4
    [N,W,H,C] = X.shape
    """
    setup variables
    """
    # 1x1 conv
    W1conv0 = tf.get_variable("W1conv0", shape=[1,1,C,MO])
    b1conv0 = tf.get_variable("b1conv0",shape=[MO])
    ##############################################################
    #
    # inception module v3
    #
    # layer | 1x1 conv + (1x1conv-3x3 conv) + [1x1-conv-3x3 conv - 3x3 conv] + [max pool-1x1 conv]
    #   C    |      4             4               4               4         = 16
    ##############################################################
    # 1x1 conv
    W1conv1 = [tf.get_variable("W1conv"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv1 = [tf.get_variable("b1conv"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    # 3x3 conv
    W3conv1= [tf.get_variable("W3conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b3conv1 = [tf.get_variable("b3conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    W3conv2 = [tf.get_variable("W3conv2"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b3conv2 = [tf.get_variable("b3conv2"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # 5x5 conv
    W5conv1= [tf.get_variable("W5conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b5conv1 = [tf.get_variable("b5conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    W5conv2 = [tf.get_variable("W5conv2"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b5conv2 = [tf.get_variable("b5conv2"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    W5conv3 = [tf.get_variable("W5conv3"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b5conv3 = [tf.get_variable("b5conv3"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # batch_normalization
    W1conv2 = [tf.get_variable("W1conv2"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv2 = [tf.get_variable("b1conv2"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    gamma = [tf.get_variable("gamma"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    beta = [tf.get_variable("beta"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    ##############################################################
    #
    #  Fully connected (affine x2)
    #
    ##############################################################
   # affine 1
    W1 = tf.get_variable("W1", shape=[int(MO/4)*int(W*H), affine_output])
    b1 = tf.get_variable("b1", shape=[affine_output]) # F
    # affine 2
    W2 = tf.get_variable("W2", shape=[affine_output,num_classes])
    b2= tf.get_variable("b2", shape=[num_classes]) # F

    """
     define graph
    """
    out = X
    ##############################################################
    #
    #  1. max pool (2x2) stride = 2
    #
    #  To reduce window size
    #
    #  INPUT : [N,W,H,C]
    # OUTPUT : [N,W/2,H/2,C]
    ##############################################################
    out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], padding='VALID')
    out = tf.nn.conv2d(out, W1conv0, strides=[1,1,1,1], padding='VALID', name='init')+b1conv0
    ##############################################################
    #
    #  2. inception_moduel
    #
    #  Concatenate result of 1x1 conv, 3x3conv, 5x5 conv, max pool
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [N,W/2, H/2, C]
    #
    ##############################################################
    for i in range(module_iter):
        # 1x1 conv
        out1 = tf.nn.conv2d(out, W1conv1[i], strides=[1,1,1,1], padding='VALID', name="one")+b1conv1[i]
        # 3x3 conv
        out3 = tf.nn.conv2d(out,W3conv1[i],strides=[1,1,1,1], padding='SAME', name = "three1")+b3conv1[i]
        out3 = tf.nn.conv2d(out3, W3conv2[i], strides = [1,1,1,1], padding='SAME',name="three2")+b3conv2[i]
        # 5x5 conv
        out5 = tf.nn.conv2d(out, W5conv1[i], strides=[1,1,1,1], padding='SAME', name="five1")+b5conv1[i]
        out5 = tf.nn.conv2d(out5, W5conv2[i], strides = [1,1,1,1], padding='SAME',name="five2")+b5conv2[i]
        out5 = tf.nn.conv2d(out5, W5conv3[i], strides = [1,1,1,1], padding='SAME',name="five3")+b5conv3[i]
        # batch_normalization
        out_batch = tf.nn.conv2d(out, W1conv2[i], strides=[1,1,1,1], padding='SAME', name="max2")+b1conv2[i]
        sample_mean, sample_var = tf.nn.moments(out_batch,[0], keep_dims=False)
        out_batch = tf.nn.batch_normalization(out_batch, sample_mean, sample_var, beta[i], gamma[i], 1e-5)
        out = tf.concat([out1, out3, out5, out_batch], 3)
        out = tf.nn.relu(out)
    ##############################################################
    #
    #  3. Fully Connected
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [#Class]
    ##############################################################
    out = tf.reshape(out,[-1,int(MO/4)*int(W*H)])
    out = tf.matmul(out,W1)+b1
    y_out = tf.matmul(out,W2)+b2
    return y_out

def ver4(X,y,is_training):
    ##############################################################
    # Change Last Affine Layer to 1x1 convolution layer
    ##############################################################
    module_iter = 10
    num_classes = 10
    MO = 64 # module output/4
    [N,W,H,C] = X.shape
    """
    setup variables
    """
    # 1x1 conv
    W1conv0 = tf.get_variable("W1conv0", shape=[1,1,C,MO])
    b1conv0 = tf.get_variable("b1conv0",shape=[MO])
    ##############################################################
    #
    # inception module v3
    #
    # layer | 1x1 conv + (1x1conv-3x3 conv) + [1x1-conv-3x3 conv - 3x3 conv] + [max pool-1x1 conv]
    #   C    |      4             4               4               4         = 16
    ##############################################################
    # 1x1 conv
    W1conv1 = [tf.get_variable("W1conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv1 = [tf.get_variable("b1conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    # 3x3 conv
    W3conv1= [tf.get_variable("W3conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b3conv1 = [tf.get_variable("b3conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    W3conv2 = [tf.get_variable("W3conv2"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b3conv2 = [tf.get_variable("b3conv2"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # 5x5 conv
    W5conv1= [tf.get_variable("W5conv1"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b5conv1 = [tf.get_variable("b5conv1"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    W5conv2 = [tf.get_variable("W5conv2"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b5conv2 = [tf.get_variable("b5conv2"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    W5conv3 = [tf.get_variable("W5conv3"+str(i), shape=[3,3,MO/4,MO/4]) for i in range(1,module_iter+1)]
    b5conv3 = [tf.get_variable("b5conv3"+str(i), shape=[MO/4]) for i in range(1,module_iter+1)]
    # max pool
    W1conv2 = [tf.get_variable("W1conv2"+str(i), shape=[1,1,MO,MO/4]) for i in range(1,module_iter+1)]
    b1conv2 = [tf.get_variable("b1conv2"+str(i), shape=[MO/4] )for i in range(1,module_iter+1)]
    ##############################################################
    #
    #  1x1 conv
    #
    ##############################################################
    W1conv3 = tf.get_variable("W1conv3", shape=[1,1,MO, MO/4])
    b1conv3 = tf.get_variable("b1conv3", shape=[MO/4]) # F
    W1conv4 = tf.get_variable("W2conv3", shape=[1,1,MO/4, num_classes])
    b1conv4 = tf.get_variable("b2conv3", shape=[num_classes]) # F

    """
     define graph
    """
    out = X
    ##############################################################
    #
    #  1. max pool (2x2) stride = 2
    #
    #  To reduce window size
    #
    #  INPUT : [N,W,H,C]
    # OUTPUT : [N,W/2,H/2,C]
    ##############################################################
    out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], padding='VALID')
    out = tf.nn.conv2d(out, W1conv0, strides=[1,1,1,1], padding='VALID', name='init')+b1conv0
    ##############################################################
    #
    #  2. inception_moduel
    #
    #  Concatenate result of 1x1 conv, 3x3conv, 5x5 conv, max pool
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [N,W/2, H/2, C]
    #
    ##############################################################
    for i in range(module_iter):
        # 1x1 conv
        out1 = tf.nn.conv2d(out, W1conv1[i], strides=[1,1,1,1], padding='VALID', name="one")+b1conv1[i]
        # 3x3 conv
        out3 = tf.nn.conv2d(out,W3conv1[i],strides=[1,1,1,1], padding='SAME', name = "three1")+b3conv1[i]
        out3 = tf.nn.conv2d(out3, W3conv2[i], strides = [1,1,1,1], padding='SAME',name="three2")+b3conv2[i]
        # 5x5 conv
        out5 = tf.nn.conv2d(out, W5conv1[i], strides=[1,1,1,1], padding='SAME', name="five1")+b5conv1[i]
        out5 = tf.nn.conv2d(out5, W5conv2[i], strides = [1,1,1,1], padding='SAME',name="five2")+b5conv2[i]
        out5 = tf.nn.conv2d(out5, W5conv3[i], strides = [1,1,1,1], padding='SAME',name="five3")+b5conv3[i]
        # max
        out_max = tf.nn.max_pool(out, [1,3,3,1], [1,1,1,1], padding='SAME',name="max1")
        out_max = tf.nn.conv2d(out_max, W1conv2[i], strides=[1,1,1,1], padding='SAME', name="max2")+b1conv2[i]
        out = tf.concat([out1, out3, out5, out_max], 3)
        out = tf.nn.relu(out)
    ##############################################################
    #
    #  3. Fully Connected(Using 1x1 conv)
    #
    #  INPUT : [N,W/2,H/2,C]
    #  OUTPUT : [#Class]
    ##############################################################
    out = tf.nn.conv2d(out, W1conv3, strides=[1,1,1,1], name="one_last1", padding='VALID')+b1conv3
    out = tf.nn.conv2d(out, W1conv4, strides=[1,1,1,1], name="one_last2", padding='VALID')+b1conv4
    out = tf.nn.avg_pool(out, [1,16,16,1], [1,1,1,1], padding='VALID')
    y_out = tf.reshape(out, [-1, num_classes])
    return y_out
