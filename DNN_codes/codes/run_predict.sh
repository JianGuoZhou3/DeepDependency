#!/bin/bash

Sample=/PATH_to_insilico_Sample/*
Model=../input/out/*pkl.gz

START=`date`

mkdir result

for i in $Sample
do
  Fold=0
  for j in $Model
  do
    Fold=$((Fold+1))
    python predict.py $j $i result/$(basename $i)"_fold_0"$Fold 
  done
done

END=`date`
echo "Start: $START"
echo "END  : $END"

rm result/input.txt result/header.txt
