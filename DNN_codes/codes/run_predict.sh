#!/bin/bash

Sample=../Clinical_sample/*
Model=./out/best_model/*pkl.gz

START=`date`

mkdir prediction

for i in $Sample
do
  Fold=0
  for j in $Model
  do
    Fold=$((Fold+1))
    python predict.py $j $i prediction/$(basename $i)"_fold_0"$Fold 
  done
done
rm ./prediction/input.txt  ./prediction/header.txt

python predict_summary.py ./prediction/

END=`date`
echo "Start: $START"
echo "END  : $END"

