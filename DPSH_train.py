#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.matlib
import scipy.io
import os
import DPSH_model

def train_net( X1, L1, U, lr, train_model):
    N = X1.shape[0]
    index = np.random.permutation(N)
    for j in xrange(N / config.batch_size + 1):
        ##random select a minibatch
        ix = index[(j * config.batch_size): min((j + 1) * config.batch_size, N)]
        S_train = calcNeighbor(L1, ix, np.arange(N))
        S_train = S_train.astype(np.float32)
        #### load image
        img = X1[ix, :, :, :]
        U[ix, :] = train_model.net['fc8'].eval(feed_dict={train_model.im_raw: img})
        train_model.train_step.run(feed_dict={train_model.im_raw: img, train_model.S: S_train, train_model.lrx: lr, train_model.Ux: U})
    return U


def test(model, dataset_L, test_L, data_set, test_data, batchsize):
    S = compute_S(dataset_L, test_L)
    B_dataset = compute_B(dataset_L, data_set, model, batchsize)
    B_test = compute_B(test_L, test_data, model, batchsize)
    map = return_map(B_dataset, B_test, S)
    return map


def compute_S(train_L, test_L):
    Dp = np.matlib.repmat(train_L, 1, len(test_L)) - np.matlib.repmat(np.transpose(test_L), len(train_L), 1)
    R = Dp == 0
    # R = 1 * R
    return R


def compute_B(L, data, train_model, batchsize):
    U = np.zeros((len(L), config.codelens))
    for j in xrange(data.shape[0] / batchsize + 1):
        img = data[j * batchsize: min((j + 1) * batchsize, data.shape[0]), :, :, :]
        features = train_model.net['fc8'].eval(feed_dict={train_model.im_raw: img})
        U[j * batchsize:min((j + 1) * batchsize, data.shape[0]), :] = features
    B_data = 1 * (U > 0)
    return B_data


def return_map(B_train, B_test, S):
    orderH = calcHammingRank(B_train, B_test)
    map = calcMAP(orderH, np.transpose(S))
    return map


def calcHammingRank(B_train, B_test):
    distH = calcHammingDist(B_test, B_train)
    orderH = np.argsort(distH, axis=1)
    return orderH


def calcHammingDist(B1, B2):
    P1 = 1 * np.sign(B1 - 0.5)
    P2 = 1 * np.sign(B2 - 0.5)

    R = P1.shape[1]
    D = np.ceil((R - P1.dot(np.transpose(P2))) / 2.0)
    return D


def calcMAP(orderH, neighbor):
    Q, N = orderH.shape
    pos = np.arange(N)
    MAP = 0
    numSucc = 0
    for i in xrange(Q):
        ngb = neighbor[i, orderH[i, :]]
        nRel = np.sum(ngb)
        if nRel > 0:
            prec = np.cumsum(ngb) / (1.0 * pos + 1.0)
            ap = np.mean(prec[ngb])
            MAP = MAP + ap
            numSucc = numSucc + 1
    MAP = MAP / (1.0 * numSucc)
    return MAP


def calcNeighbor(label, idx1, idx2):
    L1 = label[idx1]
    L2 = label[idx2]
    Dp = np.matlib.repmat(L1, 1, len(L2)) - np.matlib.repmat(np.transpose(L2), len(L1), 1)
    R = Dp == 0
    R = 1 * R
    return R

# def logExpTrick(X):
#     Y = X
#     X = np.reshape(X, [-1])
#     dx = X < 30
#
#
#     #Z = np.select(dx, np.reshape(np.log(1 + np.exp(X)), [-1]), X)
#
#     return Y


def calcNeighbor_L(L1, L2):
    Dp = np.matlib.repmat(L1, 1, len(L2)) - np.matlib.repmat(np.transpose(L2), len(L1), 1)
    R = Dp == 0
    R = 1 * R
    return R

class get_config():

  """Small config."""
  maxIter = 150
  lamda = 10.0
  lr = 0.005
  codelens = 32
  batch_size = 128

  def __init__(self, _N_size,_vgg_path):

    self.N_size = _N_size
    self.vgg_path = _vgg_path


if __name__ == "__main__":

    vgg_path = './imagenet-vgg-f.mat'
    mat = scipy.io.loadmat('cifar-10.mat')
    train_data = mat['train_data'].transpose(3, 0, 1 , 2).astype(np.float32)
    train_L = mat['train_L'].astype(np.float32)
    dataset_L = mat['dataset_L'].astype(np.float32)
    test_L = mat['test_L'].astype(np.float32)
    data_set = mat['data_set'].transpose(3, 0, 1 , 2).astype(np.float32)
    test_data = mat['test_data'].transpose(3, 0, 1 , 2).astype(np.float32)

    N_train = train_L.shape[0]
    N_train_index = np.arange(N_train)
    map_record = []
    loss_record = []
    config = get_config(N_train,vgg_path)

    U = np.zeros((train_data.shape[0], config.codelens))  # for caculating loss
    U_train = np.zeros((N_train, config.codelens))   # for updating U
    lr = config.lr
    
    gpuconfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    with tf.Graph().as_default(), tf.Session(config=gpuconfig) as session:
        train_model = DPSH_model.model(config)
        session.run(tf.initialize_all_variables())
        start = time.clock()
        for iter in xrange(config.maxIter):
            if iter % 10 == 0 and iter != 0:
                lr = lr * 0.8
            U_train = train_net(train_data, train_L, U_train, lr, train_model)

            #caculate loss
            if iter % 5 == 0 :
                S_ = calcNeighbor_L(train_L, train_L)
                for j in xrange(N_train / config.batch_size + 1):
                    ##random select a minibatch
                    ix = N_train_index[(j * config.batch_size): min((j + 1) * config.batch_size, N_train)]
                    U[ix, :] = train_model.net['fc8'].eval(feed_dict={train_model.im_raw: train_data[ix, :, :, :]})
                print iter
                theta_train = 1.0 / 2 * U.dot(np.transpose(U))
                theta_train_ = 1+np.log(np.exp(theta_train))
                print iter
                B_code = np.sign(U)
                loss_ = np.divide((- np.sum(np.multiply(S_, theta_train) - theta_train_) + config.lamda * np.sum(
                    np.power((B_code - U), 2))), float(config.N_size*config.batch_size))
                print " iter %d loss is %f,lr:%f" % (iter, loss_,lr)
                loss_record.append(loss_)
            if iter % 10 == 0:
                map = test(train_model, dataset_L, test_L, data_set, test_data, config.batch_size)
                map_record.append(map)
                print "iter %d map:%f" % (iter, map)
        plt.plot(loss_record)
        plt.show()

        end = time.clock()
        print "running time %d" % (end - start)

        map = test(train_model, dataset_L, test_L, data_set, test_data, config.batch_size)
        print "map:%f" % map


