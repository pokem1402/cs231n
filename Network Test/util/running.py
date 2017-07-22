import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import math

def cifar_10_running(model,X_train, y_train, X_val, y_val, X_test, y_test):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    y_out = model(X,y,is_training)
    optimizer = tf.train.AdamOptimizer(1e-3)
    total_loss = tf.losses.softmax_cross_entropy(logits =y_out,onehot_labels=tf.one_hot(y,10))
    mean_loss =  tf.reduce_mean(total_loss)
    train_step = optimizer.minimize(mean_loss)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    print('Training')
    run_model(X,y,is_training,mean_loss,sess,y_out,mean_loss,X_train,y_train,10,64,100,train_step,True)
    end_time = time.time()
    print (end_time-start_time)
    print('Validation')
    run_model(X,y,is_training,mean_loss,sess,y_out,mean_loss,X_val,y_val,1,64)
    print('Training')
    run_model(X,y,is_training,mean_loss,sess,y_out,mean_loss,X_train,y_train,1,64)
    print('Validation')
    run_model(X,y,is_training,mean_loss,sess,y_out,mean_loss,X_val,y_val,1,64)
    print('Test')
    run_model(X,y,is_training,mean_loss,sess,y_out,mean_loss,X_test,y_test,1,64)

def run_model(X,y,is_training,mean_loss,session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct
