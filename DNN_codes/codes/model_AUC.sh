#!/bin/bash
# selecting the best AUC model.


cd out
ln -s ../model_selection.py ./
python model_selection.py
cd ..

# Test AUC
TEST=../input/testset.tsv
Model=./out/best_model/*pkl.gz

Fold=0
for j in $Model
do
  Fold=$((Fold+1))
  python predict_model.py $j $TEST ./out/"Test_"fold_0$Fold "TEST"
done

# Training AUC
TRAIN=../input/DeepInput_fold_*
Model=./out/best_model/

Fold=0
for i in $TRAIN
do
  Fold=$((Fold+1))
  python predict_model.py $Model$(basename $i)*pkl.gz $i ./out/"Train_"fold_0$Fold "TRAIN"
done

Rscript AUC.r
rm ./out/input.txt ./out/header.txt

