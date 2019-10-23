#!/usr/bin/python
import sys,os,glob

dir=sys.argv[1]
T_cut = 0.5

samples=glob.glob(dir+"*_fold_01")

for i in samples:
	a=i.split('/')
	index= "_".join(a[-1].split('_')[:3])
	T_dic={}
	T_avg={}
	for j in range(1,6):
		file=index+"_0"+str(j)
		f=open(dir+file).readlines()
		for k in f:
			cols=k.strip().split('\t')
			gene=cols[0]
			pred=cols[1]
			if gene in T_dic:
				T_avg[gene] += float(pred)
				if float(pred) > T_cut:
					T_dic[gene] += 1
			else:
				T_avg[gene] = float(pred)
				if float(pred) > T_cut:
					T_dic[gene] = 1
				else:
					T_dic[gene] = 0

	w=open(dir+index.split('_')[0]+".pred", "w")
	w.write("Gene\tVotes\tAve.Pred\n")
	for i in T_dic:
		w.write(i+"\t"+str(T_dic[i])+"\t"+str(T_avg[i]/5)+"\n")

	w.close()
	
