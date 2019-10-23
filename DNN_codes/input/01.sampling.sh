#!/bin/bash

FILE=$1

# Shuffle
grep '1$' $FILE |shuf > temp_True
grep '0$' $FILE |shuf > temp_False

TRUE_NUM=`wc -l temp_True   | awk '{print $1}'`
FALSE_NUM=`wc -l temp_False | awk '{print $1}'`
if [ $TRUE_NUM -gt $FALSE_NUM ]
then
  DATA_NUM=$FALSE_NUM
else
  DATA_NUM=$TRUE_NUM
fi
TEST_NUM=$((DATA_NUM/5))
head -$TEST_NUM temp_True > testset.tsv
head -$TEST_NUM temp_False >> testset.tsv

TRAIN_NUM=$((DATA_NUM-TEST_NUM))
tail -$TRAIN_NUM temp_True | cut -f 2- > infile.tsv
tail -$TRAIN_NUM temp_False | cut -f 2- >> infile.tsv

echo "----------------------------"
echo "Total train :"  `wc -l infile.tsv | awk '{print $1}'` ", True samples :"   `grep '1$' infile.tsv | wc -l` ", False samples :"  `grep '0$' infile.tsv |wc -l`
echo "Total test :"  `wc -l testset.tsv | awk '{print $1}'` ", True samples :"   `grep '1$' testset.tsv | wc -l` ", False samples :"  `grep '0$' testset.tsv |wc -l`
echo "----------------------------"
rm temp_*
