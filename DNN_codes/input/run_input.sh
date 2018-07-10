#!/bin/bash
# make DNN model input 

INPUT="Model_input.txt" 
FILE_IDX="DeepInput_fold"

sh 01.sampling.sh $INPUT
python2.7 02.mkDeepInput.py $FILE_IDX


