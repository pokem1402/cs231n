import numpy as np
import time


train = np.random.randint(10, size = (50, 3027))
test = np.random.randint(10, size = (500, 3027))

def checkCorrectness(x,y):
    return bool(np.sum(x-y) == 0)

def measureTime(func, *parameters):
    t = time.time()
    ret = func(*parameters)
    elapsed = time.time()-t
    return (elapsed, ret)

# code using original L2 method
def doubleloop(train,test):
    num_train = train.shape[0]
    num_test = test.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        differ = test[i]-train[j]
        differ = np.square(differ)
        sum_differ = np.sum(differ)
        dists[i,j] = np.sqrt(sum_differ)
    return dists

# code using inner product, 1.5 times faster than original
def doubleloop_alter(train,test):
    num_train = train.shape[0]
    num_test = test.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        differ = test[i]-train[j]
        dists[i,j] = np.sqrt(np.dot(differ, differ))
    return dists

# code using original L2 method
def onceloop(train,test):
    num_train = train.shape[0]
    num_test = test.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        differ = train - test[i]
        differ = differ**2
        sum_differ = np.sum(differ, axis=1)
        dists[i] = np.sqrt(sum_differ)
    return dists

# code using inner product. is it faster than original?
def onceloop_alter(train, test):
    num_train = train.shape[0]
    num_test = test.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        differ = train - test[i]
        dif = np.sum(differ**2)
        dists[i] = np.sqrt(mtx)
    return dists

def noloop(train,test):
    num_train = train.shape[0]
    num_data = train.shape[1]
    num_test = test.shape[0]
    trn = np.sum(np.square(train), axis = 1)
    trn = np.tile(trn, (num_test,1))
    tst = np.sum(np.square(test), axis = 1)
    tst = (np.tile(tst, (num_train,1))).T
    iat = -2.*np.dot(test, train.T)
    return np.sqrt(trn+tst+iat)

(t1, dist1) = measureTime(doubleloop, train, test)
print (t1)
(t1_,dist1_) = measureTime(doubleloop_alter, train, test)
print (t1_, np.sum(np.square(dist1-dist1_)))
(t2, dist2) = measureTime(onceloop, train, test)
print (t2, np.sum(np.square(dist1-dist2)))
# (t2_, dist2_) = measureTime(onceloop_alter, train, test)
# print (t2_, np.sum(dist1-dist2_))
(t3, dist3) = measureTime(noloop, train, test)
print (t3, np.sum(np.square(dist1-dist3)))
