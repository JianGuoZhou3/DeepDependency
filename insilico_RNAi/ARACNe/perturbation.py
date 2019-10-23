#!/usr/bin/python
import os,sys
import networkx as nx
import random

NETWORK_FILE     = sys.argv[1]
BASAL_EXPRESSION = sys.argv[2]
OUTDIR           = sys.argv[3]
RESULT           = sys.argv[4]

def create_network(network_file):
	G=nx.DiGraph()
	f=open(network_file)
	for i in f:
		a=i.strip().split('\t')
		G.add_edge(a[0],a[1],corr=a[2])
	return G

def expression_data(basal_expression):
	f=open(basal_expression); ff=f.readline(); fl=f.readlines(); f.close()
	gene_dic = []
	for i in ff.strip().split('\t')[1:]:
		gene_dic.append(i)
	exp_dic = {}
	for i in fl:
		a=i.strip().split('\t')
		cell = a[0]; exp = [float(i) for i in a[1:]]; 
		new_gene_dic = [s +":" +cell for s in gene_dic]
		exp_dic.update(dict(zip(new_gene_dic,exp)))
	return exp_dic

def network_successor(G, node, cell, exp_dic, temp_exp):
	Neighbors = G.neighbors(node)
	for i in Neighbors:
		pathway = nx.shortest_path(G,source=node,target=i)
		regulator = pathway[0]
		target    = pathway[1]
		if temp_exp[target] == exp_dic[target+":"+cell] and exp_dic[regulator+":"+cell] != 0:
			delta = float(temp_exp[regulator] - exp_dic[regulator+":"+cell] ) /  exp_dic[regulator+":"+cell]
			corr = float(G.adj[regulator][target]['corr'])
			temp_exp[target] += delta * corr * exp_dic[target+":"+cell]
		else:   
			continue

def perturbation_expression(G, exp_dic, perturbed_output):
	w=open(perturbed_output,"w")
	w.write("Gene:Cell\t")	
	temp_genes = []
	for i in exp_dic:
		temp_genes.append(i.split(':')[0])
	temp_genes = sorted(set(temp_genes))
	w.write("\t".join(temp_genes)+"\n")
	for i in exp_dic:
		a=i.strip().split('\t')
		GENE = a[0].split(':')[0]
		CELL = a[0].split(':')[1]
		temp_exp	= {}
		for j in exp_dic:
			if j.split(':')[1] == CELL:
				temp_exp[j.split(':')[0]] = exp_dic[j]
		temp_exp[GENE] = temp_exp[GENE]/5	# 80% efficacy of shRNA knockdown
		network_successor(G, GENE, CELL, exp_dic, temp_exp)
		w.write(a[0])
		for k in sorted(set(temp_exp)):
			w.write("\t"+str(temp_exp[k]))
		w.write("\n")
	w.close()

def main():
	if os.path.exists(OUTDIR)==False:
		os.system("mkdir {}".format(OUTDIR))
	G=create_network(NETWORK_FILE)
	exp_dic=expression_data(BASAL_EXPRESSION)
	perturbation_expression(G, exp_dic, OUTDIR+"/"+RESULT)


if __name__ == "__main__":
	main()
