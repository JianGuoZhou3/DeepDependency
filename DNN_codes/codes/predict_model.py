import os,sys
import cPickle
import gzip
from SdA import SdA
from logistic_sgd import LogisticRegression

from Load_data import Load_data
import theano
import theano.tensor as T
import numpy
import subprocess

def relu(x):
    return T.maximum(0., x)

if len(sys.argv) != 5:
    print "\nUsage: {0} modelFile inFile predFile Type\n\n".format(sys.argv[0])
    exit(1)

modelFile	= sys.argv[1]
dataset		= sys.argv[2]
predFile	= sys.argv[3]
Type		= sys.argv[4]


def test(modelFile, dataset, predFile):
	os.system("cut -f 2- %s > %s/input.txt"  %(dataset, "/".join(predFile.split('/')[0:-1]) ) )
	os.system("cut -f 1 %s  > %s/header.txt " %(dataset, "/".join(predFile.split('/')[0:-1]) ) )
	header=[]
	f=open("%s/header.txt" %("/".join(predFile.split('/')[0:-1]))).readlines()
	for i in f:
		header.append(i.strip())
	# read model
	classifier	= cPickle.load(gzip.open(modelFile))
	# read input
	DATA = numpy.loadtxt("%s/input.txt" %("/".join(predFile.split('/')[0:-1])) )
	DATA_x = DATA[:,:-1]
	DATA_y = DATA[:,-1]
	shared_x = theano.shared(numpy.asarray(DATA_x, dtype=theano.config.floatX), borrow=True)
	layer=classifier.layers[-1]
	predict_model = theano.function( inputs = [classifier.x], 
				 outputs= layer.p_y_given_x )
	# prediction
	x_	= numpy.asarray(shared_x.get_value(borrow=True) , dtype='float32')
	y_pred 	= predict_model(x_)[:,1]
	# save result
	numpy.savetxt(predFile, zip(*[DATA_y, y_pred]) ,fmt="%s", delimiter='\t')

def train(modelFile, dataset, predFile):
	# read model
	classifier	= cPickle.load(gzip.open(modelFile))
	# read input
	datasets	= Load_data(dataset)
	test_set_x, test_set_y = datasets[2]
	layer=classifier.layers[-1]
	y     = T.ivector('y')
	predict_model = theano.function( inputs = [classifier.x, y], 
					 outputs= layer.p_y_given_x[T.arange(y.shape[0]),1] )
	get_y	= theano.function([], test_set_y)
	y_      = get_y()
	x_	= numpy.asarray(test_set_x.get_value(borrow=True) , dtype='float32')
	y_pred 	= predict_model(x_, y_)
	numpy.savetxt(predFile, zip(*[y_, y_pred]) ,fmt="%s")

def main():
	if Type =="TEST":
		test(modelFile, dataset, predFile)
	elif Type =="TRAIN":
		train(modelFile, dataset, predFile)
	else:
		print "Please specify 'TEST' or 'TRAIN'"

if __name__=="__main__":
	main()


