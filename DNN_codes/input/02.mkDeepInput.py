#!usr/bin/python
import sys,os,gzip,cPickle, numpy, time
from sklearn.model_selection import KFold

idx=sys.argv[1]

def getCnt(x):
	Cnt = dict()
	for c in x:
		Cnt[c] = Cnt.get(c, 0)+1
	return Cnt

def mkinput(inFile, outFileprefix):
	print ('Read Files...')

	DATA = numpy.loadtxt(inFile)
	print ('Data samples %d , Data features %d' % (DATA.shape[0], DATA.shape[1]-1))
	r = numpy.random.RandomState(123)
	r.shuffle(DATA)

	LastIdx =DATA.shape[1]-1
	kf1 = KFold(n_splits=5)
	num=0

	for trainIdx, testIdx in kf1.split(DATA):
		num += 1
		test_setX  = numpy.array(DATA[testIdx, 0:LastIdx] )		# DATA[row, column]
		test_setY  = numpy.array(DATA[testIdx,   LastIdx] )
		shuf_DATA  = numpy.array(DATA[trainIdx,         ] )
		r.shuffle(shuf_DATA)
		validIdx = shuf_DATA.shape[0]/4					# 1/4 valid (learning stop) 3/4 train
		
		valid_setX = numpy.array(shuf_DATA[:validIdx, 0:LastIdx] )
		valid_setY = numpy.array(shuf_DATA[:validIdx,   LastIdx] )
		train_setX = numpy.array(shuf_DATA[validIdx:, 0:LastIdx] )
		train_setY = numpy.array(shuf_DATA[validIdx:,   LastIdx] )

		cTrain = getCnt( train_setY.tolist() )
		cValid = getCnt( valid_setY.tolist() )
		cTest  = getCnt( test_setY.tolist()  )
		print ('Train set samples: ',train_setX.shape[0],  ',class: ', cTrain)
		print ('Valid set samples: ',valid_setX.shape[0], ' ,class: ', cValid)
		print ('Test  set sampels: ',test_setX.shape[0],  ' ,class: ', cTest)
		if num < 10:
		   	outFile=outFileprefix+'_0'+str(num)+'.pkl.gz'
		else:
			outFile=outFileprefix+ '_'+str(num)+'.pkl.gz'
		print ('zip data file name is '+outFile+'.pkl.gz')
		data_set= [	(train_setX, train_setY),
				(valid_setX, valid_setY),
				(test_setX,  test_setY)   ]
		cPickle.dump(data_set, gzip.open(outFile,'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
		print ('**')

mkinput("infile.tsv", idx)

