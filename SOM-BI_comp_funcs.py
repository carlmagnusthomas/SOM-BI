#!/usr/bin/env python
# coding: utf-8

# ## SOM_BIs_comp_trans.ipynb
# 
# This notebooked identifies the optimum set of node clusters to define the SOM blocking index from the SOM dataset of best matching units (SOM_data) and the ground truth dataset (GTD)
# 
# 

# In[1]:


import xarray as xr
import numpy as np
import matplotlib as mpl
mpl.use("agg", warn=False)
import matplotlib.pyplot as plt
import glob
import csv

def create_nodes_arr(row,col):
    """
    Create the arrangement of nodes to label the set
    assumes a square arrangement of nodes
    """
    arr = np.ones((row, col))
    for i in range(row):
        for j in range(col):
            arr[i,j] = arr[i,j]*(col*i+1)+j
    return arr



def plot_hist_blo(blocking_occ, SOM_nodenum, savefig_str, num_nodes, show_plots, caption=0):
    """
    Plot the histograms of blocking for the GTD and not in the GTD
    """
    alpha_val = 0.6
    nodes_notblo = SOM_nodenum.values*(blocking_occ==0).values
    nodes_blo = SOM_nodenum.values*blocking_occ.values
    freq_blo, freq_notblo = int(float(blocking_occ.mean())*1000)/10, int(float((blocking_occ==0).mean()*1000))/10
    nodes_blo_rem, nodes_notblo_rem = nodes_blo[nodes_blo != 0], nodes_notblo[nodes_notblo != 0]

    fig, ax = plt.subplots(1,1)
    ax.hist(nodes_notblo_rem-0.5, bins = np.arange(0.5,num_nodes+1.5,1), density = True, alpha = alpha_val, label = f"not blocked ({freq_notblo}% of days)", ec = "k") 
    ax.hist(nodes_blo_rem-0.5, bins = np.arange(0.5,num_nodes+1.5,1), density = True, alpha = alpha_val, label = f"blocked ({freq_blo}% of days)", ec = "k") 

    if caption != 0:
        ax.set_xlabel(f"SOM node number \n \n {caption}")
    else:
        ax.set_xlabel(f"SOM node number")
    ax.set_ylabel("Frequency of occurence")
    ax.set_xticks(np.arange(1,num_nodes+1,1))
    ax.legend()
    if show_plots == False:
        plt.close();

    fig.savefig(savefig_str, bbox_inches="tight", dpi = 300)


def calc_skill_cluster_sets(blocked_days, GTD, GTD_seas, persis_thresh, SOM_nodes, blocks_one_clusnum, skill_str, seas):
    """
    Calculate the skill score of different clusters
    """
    prec_arr, recall_arr, F1_arr, clus_num_arr = [], [], [], []

    prec_vals = sorted(np.unique(blocks_one_clusnum[skill_str].values), reverse = True)
    #loop through first element separately so that subsequent values can be appended
    node_cluster_set_test_str, ds_arr = [], []
    for prec in prec_vals:
        node_cluster_set_test_str_app = blocks_one_clusnum['set'][np.where(blocks_one_clusnum[skill_str]==prec)[0]].values
        for clus in node_cluster_set_test_str_app:
            #add cluster to cluster set
            node_cluster_set_test_str = np.append(node_cluster_set_test_str, clus)
            node_cluster_set_test_str = np.unique(node_cluster_set_test_str)
            node_num = len(node_cluster_set_test_str) # number of nodes in cluster set
            clus_num_arr.append(node_num)
            #calculate skill score of cluster set by calculating the number of days blocked from the GTD and selecting the season
            blocked_days_clus = calc_blocked_days_clus(blocked_days, persis_thresh, SOM_nodes, node_cluster_set_test_str)
            blocked_days_clus_xr = xr.DataArray(blocked_days_clus, name = "blocking", dims={"time": GTD['time']})
            blocked_days_clus_xr['time'] = GTD['time']
            blocked_days_clus_sel = blocked_days_clus_xr.sel(time=np.isin(blocked_days_clus_xr['time.season'], seas))
            prec, recall, F1 = calc_pr_rc_F1(GTD_seas, blocked_days_clus_sel)
            prec_arr.append(prec)
            recall_arr.append(recall)
            F1_arr.append(F1)

    return clus_num_arr, prec_arr, recall_arr, F1_arr

def plot_SOM_BI_skill(clus_num_arr, prec_arr, recall_arr, F1_arr, fig_str, show_plots):
    """
    Create the plots showing the skill score of the blocking patterns
    """
    fig = plt.figure()
    alpha_val = 0.7
    #plt.title(f"Skill score of different sets of node clusters \n for {nodes} nodes {var_str} vs GTD")
    plt.scatter(clus_num_arr, prec_arr, label = "precision", marker = "^", alpha = alpha_val, color = "r")
    plt.scatter(clus_num_arr, recall_arr, label = "recall", marker = "v", alpha = alpha_val, color = "g")
    plt.scatter(clus_num_arr, F1_arr, label = "F1", marker = "x", alpha = alpha_val, color = "b")
    plt.legend(bbox_to_anchor=(1,1))
    plt.xlabel("Number of node clusters selected")# \n node clusters are added to the set, starting with the nodes that \n have the highest precision")
    plt.ylabel("F1 score")
    print(f"peak F1 score = {max(F1_arr)}")
    peak_F1_idx = np.where(np.array(F1_arr) == max(F1_arr))[0][0]
    print(f"prec for peak F1 score = {prec_arr[peak_F1_idx]}")
    print(f"recall for peak F1 score = {recall_arr[peak_F1_idx]}")    
    print(f"number of clusters for peak F1 score = {clus_num_arr[peak_F1_idx]}") 
    fig.savefig(fig_str, bbox_inches="tight", dpi = 300)
    if show_plots == False:
        plt.close();


def calc_blocked_days_clus(blocked_days, persis_thresh, SOM_nodes, node_cluster_set_test_str):
    """
    Calculate the number of blocked days in the cluster
    NB this function is fairly slow - perhaps a list comprehension could speed it up?
    SOM_nodes - list of labelled SOM nodes for each day [1,2,9...,7,6]
    blocked_days - arrangement of blocked days in the GTD
    persis_thresh - threshold for number of days to limit blocking
    node_cluster_set_test_str - the set of node clusters being tested, stored as a list of strings ['[1, 2, 5]', '[1, 4]', ... , '[1]']
    """
    #define the new set of blocked days
    blocked_days_clus = np.zeros((len(blocked_days)))
    #set blocking for the five day clusters where the given SOM cluster is in the set of node clusters
    for i in range(len(blocked_days)-persis_thresh):
        SOM_cluster = str(np.unique(SOM_nodes[i:i+persis_thresh]))
        if SOM_cluster in node_cluster_set_test_str:
            blocked_days_clus[i:i+persis_thresh] = 1
    return blocked_days_clus


def calc_pr_rc_F1(GTD, block_occ):
    """
    Calculate skill scores from input DataArrays
    """
    true_pos = ((GTD + block_occ)==2).sum()
    false_pos = sum([1 if BI_val==1 and GTD[i]==0 else 0 for i, BI_val in enumerate(block_occ.values)])
    precision = float(true_pos/(true_pos+false_pos))
    recall = float(true_pos/(GTD.sum()))
    if precision == 0:
        F1 = 0
    else:
        F1 = float(2 * (precision * recall) / (precision + recall))
    
    return precision, recall, F1



def plot_SOM_BI_performance(row, col, SOM_data, skill_str, fig_str_SOM_BI_skill, fig_str_SOM_GTD_hist, GTD, var_str, seas, persis_thresh, show_plots=False):
    """
    Create plots to show the SOM-BI performance
    rowcol - the number of rows and columns in the data (will need to change when using non-square node topologies)
    """
    #skill score to order the clusters by
    skill_str = "precision"

    nodes_arr = create_nodes_arr(row, col)
    nodes = np.prod(nodes_arr.shape)
    #need to identify some way of amtching the two datasets
    #identify the number assigned to the node
    SOM_data_node_list = [(SOM_data[:,:,i]*nodes_arr).sum(axis=0).sum(axis=0) for i in range(SOM_data.shape[2])]
    SOM_nodenum = xr.concat(SOM_data_node_list, dim = "node_num")
    SOM_nodenum_seas = SOM_nodenum.isel(node_num = np.isin(GTD['time.season'], seas))
    GTD_seas = GTD.sel(time = np.isin(GTD['time.season'], seas))
    #plot the histograms of blocked and non-blocked days
    plot_hist_blo(GTD_seas, SOM_nodenum_seas, fig_str_SOM_GTD_hist, nodes, show_plots, caption=f"based on GTD and {var_str} SOMs")

    ds_arr = []
    #implement the persistence within the blocking criterea by studying every five day period
    blocked_days, SOM_nodes = GTD.values, SOM_nodenum.values
    persis_block = []
    SOM_cluster = []
    for i in range(len(blocked_days)-persis_thresh):
        if sum(blocked_days[i:i+persis_thresh]) == persis_thresh:
            persis_block.append(1)
        else:
            persis_block.append(0)
        SOM_cluster.append(str(np.unique(SOM_nodes[i:i+persis_thresh])))

    SOM_clusters_block, SOM_clusters_block_occ = np.unique([clus for i, clus in enumerate(SOM_cluster) if (persis_block[i] == 1)], return_counts=True)
    SOM_clusters_noblock, SOM_clusters_noblock_occ = np.unique([clus for i, clus in enumerate(SOM_cluster) if (persis_block[i] == 0)], return_counts=True)
    
    print("Calculating skill of individual clusters")
    blocks_one_clusnum = calc_skill_clusters(blocked_days, GTD, GTD_seas, persis_thresh, SOM_nodes, SOM_clusters_block, seas)
    
    print("Calculating skill of combined clusters")
    clus_num_arr, prec_arr, recall_arr, F1_arr = calc_skill_cluster_sets(blocked_days, GTD, GTD_seas, persis_thresh, SOM_nodes, blocks_one_clusnum, skill_str, seas)
    plot_SOM_BI_skill(clus_num_arr, prec_arr, recall_arr, F1_arr, fig_str_SOM_BI_skill, show_plots)
    
    #calculate the optimal set of node clusters
    #defined here as the optimal F1 score
    #another possibility is the crossing point of precision and recall values 
    #NB this isn't necessarily the best cluster set in terms of F1 score,
    #or even the best set in general, but it is probably quite close to the optimal value
    diff_prec_recall = abs(np.array(prec_arr)-np.array(recall_arr))
    optimal_idx_cross = np.where(diff_prec_recall == min(diff_prec_recall))[0][0]
    optimal_idx_F1 = np.where(np.array(F1_arr) == max(F1_arr))[0][0]      
    #define optimal values using the cluster set that maximises F1 score
    optimal_prec, optimal_recall, optimal_F1 = prec_arr[optimal_idx_F1], recall_arr[optimal_idx_F1], F1_arr[optimal_idx_F1]
    
    #order elements from best - worst precision
    blocks_one_clusnum_sort_prec = blocks_one_clusnum.sortby("precision", ascending=False)
    best_cluster_set_cross = blocks_one_clusnum_sort_prec['set'][:optimal_idx_cross+1]
    best_cluster_set_F1 = blocks_one_clusnum_sort_prec['set'][:optimal_idx_F1+1]
    
    return SOM_clusters_block, blocked_days, persis_thresh, SOM_nodes, best_cluster_set_cross, best_cluster_set_F1, optimal_prec, optimal_recall, optimal_F1

def calc_skill_clusters(blocked_days, GTD, GTD_seas, persis_thresh, SOM_nodes, SOM_clusters_block, seas):
    """
    Calculates the individual skill score of each cluster
    takes several seconds because ofthe creation of the blocked clusters
    """
    ds_arr_ones = []
    for clus in SOM_clusters_block:
        node_cluster_set_test = [clus]
        node_cluster_set_test_str = [str(clus).replace(',', '') for clus in node_cluster_set_test]
        #calculate the blocked days which the new cluster determines
        blocked_days_clus = calc_blocked_days_clus(blocked_days, persis_thresh, SOM_nodes, node_cluster_set_test_str)
        #define as DataArray and select JJA to remove the extended days included for classifying blocks
        blocked_days_clus_xr = xr.DataArray(blocked_days_clus, name = "blocking", dims={"time": GTD['time']})
        blocked_days_clus_xr['time'] = GTD['time']
        blocked_days_clus_seas = blocked_days_clus_xr.sel(time=np.isin(blocked_days_clus_xr['time.season'], seas))
        prec, recall, F1 = calc_pr_rc_F1(GTD_seas, blocked_days_clus_seas)
        #calculate precision, recall and F1
        if len(str(node_cluster_set_test)) == 1:
            comb_str = f"{node_cluster_set_test[0]}".replace("'", "")
        else:
            comb_str = f"{str(node_cluster_set_test)[1:-1]}".replace("'", "")        
        ds=xr.Dataset({'precision': prec, 'recall': recall, 'F1': F1, 'clus_num': int(len(node_cluster_set_test)), 'set': str(comb_str)})
        ds_arr_ones.append(ds)
    blocks_one_clusnum = xr.concat(ds_arr_ones, dim = "set")
    return blocks_one_clusnum

def calc_F1_cv(SOM_data_cv, GTD_cv, best_cluster_set, persis_thresh, nodes_arr, seas):
    """
    using the SOM BI derived from the training data
    create a classification of blocking events and calculate the F1_cv
    This is used to calculate the generalisation score
    """
    blocked_days = GTD_cv.values
    SOM_data_node_list_cv = [(SOM_data_cv[:,:,i]*nodes_arr).sum(axis=0).sum(axis=0) for i in range(SOM_data_cv.shape[2])]
    SOM_nodenum_cv = xr.concat(SOM_data_node_list_cv, dim = "node_num")    
    
    blocked_days_clus = calc_blocked_days_clus(blocked_days, persis_thresh, SOM_nodenum_cv, best_cluster_set)
    blocked_days_clus_xr = xr.DataArray(blocked_days_clus, name = "blocking", dims = {"time": GTD_cv['time']})
    blocked_days_clus_xr['time'] = GTD_cv['time']
    blocked_days_clus_sel = blocked_days_clus_xr.sel(time = np.isin(blocked_days_clus_xr['time.season'], seas))
    GTD_cv_seas = GTD_cv.sel(time = np.isin(blocked_days_clus_xr['time.season'], seas))
    prec, recall, F1 = calc_pr_rc_F1(GTD_cv_seas, blocked_days_clus_sel)   
    if F1 == np.nan:
        F1 = 0
    return F1, prec, recall

