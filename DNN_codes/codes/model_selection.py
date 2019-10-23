#!/usr/bin/python
import os, sys

# model list
os.system("ls *log > temp.txt ; sed 's/DeepInput_fold_0[1-5].pkl.gz_//g' temp.txt |sort -u > out.list")


infile="out.list"
f=open(infile).readlines()
dic=[]
for i in f:
	dic.append(i.strip())
	
w=open("summary.txt", "w")
w.write("ave.AUC\tave.Sensitivity\tave.Specificity\tModel_parameters\n")
for i in dic:
	fold1="DeepInput_fold_01.pkl.gz_"+i
	fold2="DeepInput_fold_02.pkl.gz_"+i
	fold3="DeepInput_fold_03.pkl.gz_"+i
	fold4="DeepInput_fold_04.pkl.gz_"+i
	fold5="DeepInput_fold_05.pkl.gz_"+i
	AUC_dic = []
	sens_dic = []
	spec_dic = []
	if os.path.isfile(fold5) == True:
		for j in (fold1, fold2, fold3, fold4, fold5):
			temp_f=open(j).readlines()
			AUC = 0 ; sens = 0 ; spec = 0
			for k in temp_f:
				cols= k.strip().split(' ')
				if cols[0] == "test":
					if float(cols[11]) > AUC:
						AUC = float(cols[11])
						sens = float(cols[5])
						spec = float(cols[8])
			AUC_dic.append(AUC)
			sens_dic.append(sens)
			spec_dic.append(spec)
		ave_AUC  = sum(AUC_dic) / float(len(AUC_dic))
		ave_sens = sum(sens_dic) / float(len(sens_dic))
		ave_spec = sum(spec_dic) / float(len(spec_dic))
		w.write("AUC:"+str(ave_AUC)+"\tSensitivity:"+str(ave_sens)+"\tSpecificity:"+str(ave_spec)+"\t"+i+"\n")
w.close()

os.system("sort -k1Vr summary.txt > temp.txt ; mv temp.txt summary.txt; \
		echo 'Best model AUC' ; cut -f 1 summary.txt |head -n 2 | tail -n 1;	\
		head -n 2 summary.txt |tail -n 1 | cut -f 4 > temp.txt; mkdir best_model models; \
		sed 's/^/mv */g' temp.txt | sed 's/.log$/* best_model/g' > temp.sh ; sh temp.sh; \
		mv DeepInput* models;  rm temp*\
")

