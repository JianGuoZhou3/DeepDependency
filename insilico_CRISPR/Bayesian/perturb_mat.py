#!/usr/bin/python
# python version 2.7.14
# numpy version 1.14.2
# pandas version 0.22.0

import numpy as np
import pandas as pd
import argparse


def parse_arguements():
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--basal", required=True, type=str,
                        help="basal expression file")
    parser.add_argument("-n", "--network", required=True, type=str,
                        help='network file')
    parser.add_argument("-o", "--outdir", type=str,
                        help="perturbed output file")

    args = parser.parse_args()
    return args


def import_network(network):
    network_df = pd.read_csv(network, sep='\t', header=0, index_col=0)
    # network file validity check
    assert set(network_df.columns) == set(network_df.index), "Network source and target genes are not identical."
    if not network_df.index.equals(network_df.columns):
        print "Warning: network source & target are not in same order."
        network_df = network_df.reindex(index=network_df.columns)
    tmp_mat = network_df.values
    np.fill_diagonal(tmp_mat, 1)
    return pd.DataFrame(data=tmp_mat, index=network_df.index, columns=network_df.columns)


def reorder_basal(header, network_index):
    # reorder basal index to mactch network index
    basal_series = pd.Series(header)
    tmp_dic = pd.Series(network_index).to_dict()
    bDict = {val: key for (key, val) in tmp_dic.iteritems()}
    idx_series = basal_series.apply(lambda x: bDict[x])
    reindex = pd.Series(idx_series.index.values, index=idx_series).sort_index().values
    return reindex


def perturb(network_mat, basal_vec):
    result = (-1 * np.matmul(network_mat, np.diag(basal_vec))) + basal_vec
    result[np.where(basal_vec == 0)[0], ] = basal_vec
    return np.round(result, decimals=5)


def main():
    args = parse_arguements()
    network_df = import_network(args.network)

    basal_file = open(args.basal, 'r')
    header = basal_file.readline().strip().split('\t')[1:]
    assert set(header) == set(network_df.index), "Basal expression genes and network genes are not identical."
    reindex = reorder_basal(header, network_df.index)

    while True:
        line = basal_file.readline()
        if not line:
            break
        tokens = line.strip().split('\t')
        cell_line = tokens[0]
        basal_vec = np.array([float(i) for i in tokens[1:]])[reindex]

        print "Processing " + cell_line
        perturbed_mat = perturb(network_df.values, basal_vec)
        new_idx = network_df.index.str.cat([cell_line] * len(network_df.index), sep=':')
        perturbed_df = pd.DataFrame(data=perturbed_mat, index=new_idx)
        perturbed_df.to_csv("{0}/{1}.tmp".format(args.outdir, cell_line), sep='\t', header=False)

    basal_file.close()


if __name__ == "__main__":
    main()
