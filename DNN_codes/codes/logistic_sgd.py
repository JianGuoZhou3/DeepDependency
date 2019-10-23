#!/usr/bin/python
#-*-coding:utf-8-*-

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from theano import gof, config

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):
	if W is None:
            W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),name='W',borrow=True
            )

	if b is None:
            b = theano.shared(
                value=numpy.zeros((n_out,),
                    dtype=theano.config.floatX
                ),name='b',borrow=True
            )
	self.W = W
	self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.x = input


    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def error(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            error_ = T.neq(self.y_pred, y)
            return T.sum(error_), error_.shape[0]
        else:
            raise NotImplementedError()
    def sensitivity(self, y):
        total_positive = T.eq( y, 1)                             # total real positive 
        positive_pred  = self.y_pred[ total_positive.nonzero() ] # predicted classes of real positive
                                                                 # ( true positive + false negative ) 
                                                                 # .nonzero() : index
        true_positive  = T.eq(positive_pred, 1)                  # true positive
        return T.sum(true_positive), T.sum(total_positive)

    def specificity(self, y):
        total_negative = T.eq( y, 0)                             # total real negative 
        negative_pred  = self.y_pred[ total_negative.nonzero() ] # predicted classes of real negative
                                                                 # ( true negative + false positive ) 
                                                                 # .nonzero() : index
        true_negative  = T.eq( negative_pred, 0)                 # true negative
        return T.sum(true_negative), T.sum(total_negative)

    def class1_pred(self, y):
        return self.p_y_given_x[T.arange(y.shape[0]),1]          # probability of class 1

