#!/bin/bash
# make DNN model input 

INPUT="Model_input.txt" 
FILE_IDX="DeepInput_fold"

bash 01.sampling.sh $INPUT
python 02.mkDeepInput.py $FILE_IDX


