import os, sys
import timeit
import gzip, cPickle, glob

import numpy
from collections import OrderedDict

import theano
import theano.tensor as T

from Load_data import Load_data
from logistic_sgd import LogisticRegression
from dA import dA
from mlp import HiddenLayer, _dropout_from_layer, DropoutHiddenLayer



def tee_stdout(f_out, msg):
    f_out.write(msg)
    f_out.write('\n')
    fout.flush()
    print(msg)

def relu(x):
    return T.maximum(0., x)

class SdA(object):
    def __init__(self, numpy_rng=None, useRelu=None, W_distribution=None, LayerNodes=None, dropout=None):

	self.n_layers = ( len(LayerNodes)-2 )
        self.dA_layers = []
	self.dropout_layers = []
        self.layers = []
        self.x = T.matrix('x') 
        self.y = T.ivector('y')
	next_layer_input = self.x
	next_dropout_layer_input = _dropout_from_layer(numpy_rng, self.x, p=dropout[0])

	weight_matrix_sizes = zip(LayerNodes, LayerNodes[1:])
	layer_counter =0

        for n_in, n_out in weight_matrix_sizes[:-1]:
            if useRelu ==True:
                activation = relu
                activation1= relu
                if layer_counter ==0:
                    activation2 = T.nnet.sigmoid
                else:
                    activation2 = T.nnet.softplus
            else:
                activation = T.nnet.sigmoid
                activation1= T.nnet.sigmoid
                activation2= T.nnet.sigmoid

            W_bound =4.*numpy.sqrt(6./(n_in+n_out))

            next_dropout_layer = DropoutHiddenLayer(numpy_rng      =numpy_rng,
                                                    input          =next_dropout_layer_input,
						    activation     =activation,
						    n_in           =n_in,
                                                    n_out          =n_out, 
                                                    W_distribution =W_distribution,
						    W_bound        =W_bound,
                                                    dropout_rate   =dropout[layer_counter+1]) 
            self.dropout_layers.append( next_dropout_layer ) 
            next_dropout_layer_input  = next_dropout_layer.output 

            next_layer = HiddenLayer(numpy_rng  =numpy_rng,
                                     input      =next_layer_input,
				     activation =activation,
				     n_in       =n_in,
                                     n_out      =n_out,
				     W          =next_dropout_layer.W * (1- dropout[layer_counter]),
				     b          =next_dropout_layer.b)
            self.layers.append(next_layer)

            dA_layer = dA(numpy_rng    =numpy_rng,
                          input        =next_layer_input,
			  useRelu      =useRelu,
                          activation1  =activation1,
                          activation2  =activation2,
                          n_visible    =n_in,
                          n_hidden     =n_out,
			  W            =next_dropout_layer.W,
			  b            =next_dropout_layer.b )
            self.dA_layers.append(dA_layer)
            next_layer_input = next_layer.output

            if layer_counter == 0:
                self.L1 = abs(next_dropout_layer.W).sum()
                self.L2 = (next_dropout_layer.W **2).sum()
            else:
                self.L1 = self.L1 + abs(next_dropout_layer.W).sum()
                self.L2 = self.L2 + (next_dropout_layer.W **2).sum()
            layer_counter +=1

	n_in, n_out = weight_matrix_sizes[-1]
	dropout_output_layer = LogisticRegression(input =next_dropout_layer_input,
						  n_in  =n_in,
                                                  n_out =n_out)

	self.dropout_layers.append( dropout_output_layer )
        self.L1 =  self.L1 + abs(dropout_output_layer.W).sum()
        self.L2 =  self.L2 + (dropout_output_layer.W **2).sum()

        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood(self.y)
	output_layer = LogisticRegression( input =next_layer_input,
                                           n_in  =n_in,
                                           n_out =n_out,
					   W     =dropout_output_layer.W *(1-dropout[-1]),
					   b     =dropout_output_layer.b )

	self.layers.append( output_layer )
	self.error    = self.layers[-1].error(self.y)
        self.sensitivity = self.layers[-1].sensitivity(self.y)
        self.specificity = self.layers[-1].specificity(self.y)
        self.class1_pred = self.layers[-1].class1_pred(self.y) 
	self.params = [ param for layer in self.dropout_layers for param in layer.params ]
	
    def pretraining_functions(self, train_set_x, batch_size):
        index         = T.lvector('index')
	corruption    = T.scalar('corruption')
	learning_rate = T.scalar('learning_rate')
        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption, learning_rate)
            fn = theano.function(
                	inputs =[index, corruption, learning_rate],
                	outputs=[cost],
                	updates=updates,
                	givens= {self.x: train_set_x[index]})
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate ,L1_param, L2_param, mom):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y)   = datasets[2]
        index = T.lvector('index')

        gparams = T.grad(self.dropout_negative_log_likelihood+L1_param*self.L1+L2_param*self.L2 , self.params )

        self.gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
            self.gparams_mom.append(gparam_mom)

	updates1 = OrderedDict()
	for param, gparam, gparam_mom in zip(self.params, gparams, self.gparams_mom):
	    updates1[gparam_mom] = mom * gparam_mom - learning_rate * gparam
	    updates1[param] = param + updates1[gparam_mom]

        train_model = theano.function(
            inputs =[index],
            outputs=self.dropout_negative_log_likelihood,
            updates=updates1,
            givens ={self.x: train_set_x[index],
                     self.y: train_set_y[index]} )
        # error check
        train_error_fn = theano.function(
            inputs = [index],
            outputs= self.error,
            givens = {self.x: train_set_x[index],
                      self.y: train_set_y[index]} )
        valid_error_fn = theano.function(
            inputs =[index],
            outputs= self.error,
            givens ={self.x: valid_set_x[index],
                     self.y: valid_set_y[index]} )
        # performance check : error rate, sensitivity, specificity, auc
        test_error_fn = theano.function(
            inputs =[index],
            outputs= self.error,
            givens ={self.x: test_set_x[index],
                     self.y: test_set_y[index]} )
        test_sensitivity_fn = theano.function(
            inputs = [index],
            outputs=self.sensitivity,
            givens ={self.x: test_set_x[index],
                     self.y: test_set_y[index]} )
        test_specificity_fn = theano.function(
            inputs = [index],
            outputs=self.specificity,
            givens ={self.x: test_set_x[index],
                     self.y: test_set_y[index]} )
        test_class1_pred_fn = theano.function(
            inputs = [index],
            outputs=self.class1_pred,
            givens ={self.x: test_set_x[index],
                     self.y: test_set_y[index]} )
        test_y_fn = theano.function(
            inputs = [index],
            outputs= self.y,
            givens ={self.y: test_set_y[index] } )

        n_train_exp = train_set_x.get_value(borrow=True).shape[0]
        n_valid_exp = valid_set_x.get_value(borrow=True).shape[0]
        n_test_exp  = test_set_x.get_value(borrow=True).shape[0]

        def getSums( fn, n_exp, batch_size ):
            val_sum = 0.
            tot_len = 0.
            n_batches = n_exp/ batch_size
            resid     = n_exp  - (n_batches * batch_size)
            IDX       = range(n_exp)
            for i in range(n_batches):
                sum_val, len_val= fn(IDX[i*batch_size:(i+1)*batch_size])
                val_sum += sum_val
                tot_len += len_val
            if resid !=0:
                sum_val, len_val= fn(IDX[n_batches*batch_size:(n_batches*batch_size)+resid])
                val_sum += sum_val
                tot_len += len_val
            return val_sum/tot_len

        def getVals( fn, n_exp, batch_size ):
            vals = list()
            n_batches = n_exp/ batch_size
            resid     = n_exp  - (n_batches * batch_size)
            IDX       = range(n_exp)
            for i in range(n_batches):
                vals+= fn(IDX[i*batch_size:(i+1)*batch_size]).tolist()
            if resid !=0:
                vals+= fn(IDX[n_batches*batch_size:(n_batches*batch_size)+resid]).tolist()
            return vals

        def errorcheck():
            train_error = getSums( train_error_fn, n_train_exp, batch_size ) 
            valid_error = getSums( valid_error_fn, n_valid_exp, batch_size ) 
            return  train_error , valid_error 

        def performance():
            test_error       = getSums(test_error_fn, n_test_exp, batch_size)
            test_sensitivity = getSums( test_sensitivity_fn, n_test_exp, batch_size )
            test_specificity = getSums( test_specificity_fn, n_test_exp, batch_size ) 
            test_y           = getVals( test_y_fn, n_test_exp, batch_size )
            test_class1_pred = getVals( test_class1_pred_fn, n_test_exp, batch_size )
            test_roc         = ROCData( zip( test_y, test_class1_pred ) )
            return test_error, test_sensitivity, test_specificity, test_roc
        return train_model, errorcheck, performance

def test_SdA(dataset=None, batch_size=None,
             pretrain_lr=None,   pretrain_epochs=None,
             finetune_lr=None,   training_epochs=None,
	     L1_param=None,      L2_param=None,
             mom=None,useRelu=None, W_distribution=None,
             dropout=None,       LayerNodes=None):

    datasets = Load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x,  test_set_y  = datasets[2]

    n_train_exp = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = n_train_exp / batch_size

    numpy_rng = numpy.random.RandomState(123)
    tee_stdout( fout, ('... building the model'))

    sda = SdA(numpy_rng      =numpy_rng,
              useRelu        =useRelu,
              W_distribution =W_distribution,
              dropout        =dropout,
              LayerNodes     =LayerNodes)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)
    tee_stdout( fout, ('... pre-training the model'))
    start_time = timeit.default_timer()

    for i in range(sda.n_layers):
 	dropout_rate = dropout[i]
        tee_stdout( fout, ('... layer %i, corruption rate %.2f , learning rate %.5f' %(i, dropout_rate, pretrain_lr)))
        epoch =0
        while ( epoch < pretrain_epochs ):
            epoch+=1
            permutation = numpy.random.permutation(n_train_exp)
            c = []
            for batch_index in range(n_train_batches):
                batch_begin = batch_index * batch_size
                batch_end   = batch_begin + batch_size
                cost_out = pretraining_fns[i](index        = permutation[batch_begin:batch_end], 
                                              corruption   = dropout_rate, 
                                              learning_rate= pretrain_lr)
                c.append(cost_out)
            this_validation_loss = numpy.mean(c)
            tee_stdout( fout, ('Pre-training layer {}, epoch {}, cost {}'.format( i, epoch, this_validation_loss)))
 
    end_time = timeit.default_timer()
    tee_stdout( fout, ('The pretraining code for file ran for %.2fm' % ((end_time - start_time) / 60.)))

    ########################
    # FINETUNING THE MODEL #
    ########################

    train_model, error_check, test_performance = sda.build_finetune_functions(
        datasets     =datasets,
        batch_size   =batch_size,
        learning_rate=finetune_lr,
        L1_param     =L1_param,
        L2_param     =L2_param,
	mom          =mom    )

    patience = 100 * n_train_batches
    patience_increase = 2.
    improvement_threshold = 0.99
    validation_frequency = min(n_train_batches, patience / 2)

    best_valid_loss = numpy.inf
#   test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    train_ACC,valid_ACC,EPOCH                          =[[] for i in range(3)]
    test_ACC, test_SPE, test_SEN, test_AUC, test_EPOCH =[[] for i in range(5)]

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        permutation = numpy.random.permutation(n_train_exp)

        for minibatch_index in range(n_train_batches):

            batch_begin = minibatch_index * batch_size
            batch_end   = batch_begin + batch_size

            train_likelihood = train_model(permutation[batch_begin:batch_end])

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:		# each epoch.
                train_error, this_validation_loss = error_check()
                train_accuracy = 1.- train_error
                valid_accuracy = 1.- this_validation_loss
                msg='epoch %i, minibatch %i/%i, validation accuracy  %.2f %%' % ( 
                        epoch, minibatch_index + 1, n_train_batches, valid_accuracy *100. )
                tee_stdout(fout, msg) 
               
                EPOCH.append(epoch)
                train_ACC.append( train_accuracy *100.)
                valid_ACC.append( valid_accuracy *100.)

                if this_validation_loss < best_valid_loss:
                    if ( this_validation_loss < best_valid_loss * improvement_threshold ):
                        patience = max(patience, iter * patience_increase)
                    best_valid_loss = this_validation_loss
                    best_iter = iter

                    test_error, test_sensitivity, test_specificity, test_roc  = test_performance()
                    test_accuracy = 1. - test_error
                    msg='\ttest accuracy %.2f %%, sensitivity %.2f %%, specificity %.2f %% AUC %.2f %%'%(
                           test_accuracy *100. , test_sensitivity *100., test_specificity *100., test_roc.auc() *100.)
		    tee_stdout(fout, msg) 

                    test_EPOCH.append(epoch)
                    test_ACC.append( test_accuracy *100. )
                    test_SPE.append( test_specificity *100. )
                    test_SEN.append( test_sensitivity *100. ) 
                    test_AUC.append( test_roc.auc() *100. )

		    # save model
                    cPickle.dump(sda, gzip.open(saveFile+'.pkl.gz', 'wb'),protocol=2)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    msg= 'Optimization complete. \nBest validation score of %f %% obtained at iteration %i\nThe code for file ran for %.2fm' % (best_valid_loss * 100., best_iter + 1, (end_time - start_time)/ 60.)
    tee_stdout( fout, msg)

if __name__ == '__main__':
######## Parameter settings..
    if os.path.exists('./out')==False:
        os.system('mkdir out')

    datasets= ["../input/DeepInput_fold_%.2d.pkl.gz" %i for i in range(1,6)]

    n_layer     = numpy.random.choice([2, 3, 4])
    Hidden      = sorted([numpy.random.choice([100, 500, 1000]) for i in range(n_layer)], reverse=True)
    dropout     = [numpy.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5]) for i in range(n_layer+1)]
    pretrain_lr = numpy.random.choice([0.1, 0.01, 0.001, 0.0001])
    finetune_lr = numpy.random.choice([0.1, 0.01, 0.001, 0.0001])
    L1_param    = numpy.random.choice([1e-08, 1e-04, 1e-02, 1e-01, 1, 10])
    L2_param    = numpy.random.choice([1e-08, 1e-04, 1e-02, 1e-01, 1, 10])
    mom         = numpy.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5])
    batch_size  = numpy.random.choice([10, 50, 100])
    useRelu	= numpy.random.choice([True, False])
    W_distribution = numpy.random.choice(["norm", "uni"])
    insize	= 1000

######## 
    for dataset in datasets:
            TMP=[str(s) for s in [ dataset.split('/')[-1], 'finetuneLR', finetune_lr,
                                   'L1Param', L1_param, 'L2Param', L2_param, 'mom', mom ]]
            TMP +=[str(s) for s in ['layer',      ','.join([str(s) for s in Hidden]),
                                    'pretrainLR',     pretrain_lr,
                                    'useRelu',        useRelu,
                                    'W_dist',         W_distribution,
				    'batch_size',     batch_size ]]
            TMP+=['dropout', ','.join([str(s) for s in dropout])]
            saveFile='out/'+'_'.join(TMP)
            fout=open(saveFile+'.log','w')
            tee_stdout( fout, ('Write file ...'+saveFile))
            tee_stdout( fout, ('Start ...'     +dataset))
######## Run ..
            test_SdA( dataset            =dataset,
                      batch_size         =batch_size,
                      pretrain_lr        =pretrain_lr,
                      pretrain_epochs    =100,
                      finetune_lr        =finetune_lr,
                      training_epochs    =10000,
                      L1_param           =L1_param,
                      L2_param           =L2_param,
                      mom                =mom,
                      useRelu            =useRelu,
                      W_distribution     =W_distribution,
                      dropout            =dropout,
                      LayerNodes         =[insize]+Hidden+[2] ) # Input/Hidden/Output
            fout.close()
