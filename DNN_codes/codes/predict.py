import os,sys
import cPickle
import gzip
from SdA import SdA
from logistic_sgd import LogisticRegression

import theano
import theano.tensor as T
import numpy
import subprocess

def relu(x):
    return T.maximum(0., x)

if len(sys.argv) != 4:
    print "\nUsage: {0} modelFile inFile predFile\n\n".format(sys.argv[0])
    exit(1)

modelFile	= sys.argv[1]
dataset		= sys.argv[2]
predFile	= sys.argv[3]

os.system("cut -f 2- %s | sed '1d' > %s/input.txt" %(dataset, "/".join(predFile.split('/')[0:-1]) ) )
os.system("cut -f 1 %s |sed '1d' >  %s/header.txt " %(dataset, "/".join(predFile.split('/')[0:-1]) ) )

header=[]
f=open("%s/header.txt" %("/".join(predFile.split('/')[0:-1]))).readlines()
for i in f:
	header.append(i.strip())


# read model
classifier	= cPickle.load(gzip.open(modelFile))

# read input
DATA = numpy.loadtxt("%s/input.txt" %("/".join(predFile.split('/')[0:-1])) )
shared_x = theano.shared(numpy.asarray(DATA, dtype=theano.config.floatX), borrow=True)
layer=classifier.layers[-1]
predict_model = theano.function( inputs = [classifier.x], 
			 outputs= layer.p_y_given_x )
# prediction
x_	= numpy.asarray(shared_x.get_value(borrow=True) , dtype='float32')
y_pred 	= predict_model(x_)[:,1]

# save result
numpy.savetxt(predFile, zip(*[header, y_pred]) ,fmt="%s", delimiter='\t')


