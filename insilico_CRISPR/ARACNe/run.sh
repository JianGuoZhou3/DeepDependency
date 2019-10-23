#!/bin/bash

NETWORK=./ARACNe_sample/ARACNe_real_breast.tsv
BASAL=./ARACNe_sample/Cellline_basal_ARACNe.txt
OUTDIR=./output
RESULT="perturbed_result.txt"

python perturbation.py $NETWORK $BASAL $OUTDIR $RESULT
