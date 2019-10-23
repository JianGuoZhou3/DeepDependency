#!/bin/bash

NETWORK=./Bayesian_sample/Bayesian_real_breast.tsv
BASAL=./Bayesian_sample/Cellline_basal_Bayesian.txt
OUTDIR=./output
RESULT="perturbed_result.txt"

# make output directory
mkdir $OUTDIR
# run
python perturb_mat.py --network ${NETWORK} --basal ${BASAL} --outdir ${OUTDIR}

# header line
head -1 ${NETWORK} > ${OUTDIR}/${RESULT}

# merge output
for TMP_FILE in $(ls ${OUTDIR} | grep .tmp)
do
    cat ${OUTDIR}/${TMP_FILE} >> ${OUTDIR}/${RESULT}
done
