import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict

class dA(object):
    def __init__( self, numpy_rng, input, useRelu, activation1, activation2, n_visible, n_hidden,
                  W=None, b=None, bvis=None ):
	self.x          = input
        self.rng        = numpy_rng
        self.useRelu    = useRelu
        self.activation1= activation1
        self.activation2= activation2

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ), borrow=True )
        self.W = W
        self.b = b
	self.b_prime = bvis
        self.W_prime = self.W.T   # Transpose
        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):  #dropout
	theano_rng = RandomStreams(self.rng.randint(2 ** 30))
	return theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return self.activation1(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return self.activation2(T.dot(hidden, self.W_prime)+ self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
	tilde_x = self.get_corrupted_input(self.x, corruption_level)
	y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        if self.useRelu == True:
            cost = T.mean( T.sum((self.x - z)**2, axis=1) )
        else:
            cost = T.mean(-T.sum(self.x * T.log(z) + (1-self.x) * T.log(1-z) , axis=1) ) 
        gparams = T.grad(cost, self.params)
	updates = OrderedDict()
	for gparam, param in zip(gparams, self.params):
	    updates[param] = param - gparam * learning_rate
        return cost, updates
