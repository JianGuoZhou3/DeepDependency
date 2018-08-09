#!/bin/bash

# Number of nodes for parallel running
MAX_J=2
TOT_J=10
START=`date`

mkdir log err
# iteration of model generation
for i in $(seq 1 $TOT_J) 
do
  OMP_NUM_THREADS=1 THEANO_FLAGS=device=cpu python SdA.py $i 1>log/log$i 2>err/err$i &
    while (true)
    do
      NUM_J=`jobs -l|wc -l`
      if [ $NUM_J -lt $MAX_J ]
      then
         break
      fi
      sleep 2
    done 
done

WORK_PID=`jobs -l |awk '{print $2}'`
wait $WORK_PID

END=`date`
echo "Start: $START"
echo "END  : $END"
