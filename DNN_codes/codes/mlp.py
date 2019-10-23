import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class HiddenLayer(object):
    def __init__(self,numpy_rng=None, input=None, n_in=None, n_out=None, activation=None, 
                 W_distribution=None, W_bound=None, W=None, b=None):
        self.input = input
        if W is None:
            if  W_distribution =='uni':
                W_values = numpy.asarray(
                    numpy_rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)
                    ),dtype=theano.config.floatX )
            elif W_distribution == 'norm':
                W_values = numpy.asarray(
                    W_bound * numpy_rng.standard_normal(size=(n_in, n_out)
                    ),dtype=theano.config.floatX )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        lin_output = T.dot(input, self.W) + self.b
	self.output = activation(lin_output)

def _dropout_from_layer( numpy_rng, input, p):
    srng = RandomStreams(numpy_rng.randint(2 **30))
    mask = srng.binomial(size=input.shape, n=1, p=1-p, dtype=theano.config.floatX)
    return input *mask

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, numpy_rng, input, n_in, n_out,activation, W_distribution, 
                 W_bound=None, W=None, b=None, dropout_rate=None ):
        super(DropoutHiddenLayer, self).__init__(numpy_rng=numpy_rng, 
                                                 input=input, n_in=n_in, n_out=n_out,
                                                 W_distribution=W_distribution,
                                                 W_bound=W_bound,activation=activation, W=W, b=b)
        self.output = _dropout_from_layer(numpy_rng, self.output, p=dropout_rate)
