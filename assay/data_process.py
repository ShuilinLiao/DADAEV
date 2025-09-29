import os
import anndata
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
from numpy.random import choice
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import random
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_simulated_data(sc_data, 
                            d_prior=None,
                            n=500, samplenum=5000,
                            random_state=None, sparse=True, sparse_prob=0.5,
                            rare=False, rare_percentage=0.4):

    print('Reading single-cell dataset, this may take 1 min')
    if '.txt' in sc_data:
        sc_data = pd.read_csv(sc_data, index_col=0, sep='\t')
        sc_data.dropna(inplace=True)
        sc_data['celltype'] = sc_data.index
        print(sc_data.index)
        sc_data.index = range(len(sc_data))

    num_celltype = len(sc_data['celltype'].value_counts())
    genename = sc_data.columns[:-1]

    celltype_groups = sc_data.groupby('celltype').groups
    sc_data.drop(columns='celltype', inplace=True)
    sc_data = anndata.AnnData(sc_data)
    sc_data = sc_data.X
    sc_data = np.ascontiguousarray(sc_data, dtype=np.float32)

    # make random cell proportions
    if random_state is not None and isinstance(random_state, int):
        print('You specified a random state, which will improve the reproducibility.')

    if d_prior is None:
        print('Generating cell fractions using Dirichlet distribution without prior info (actually random)')
        if isinstance(random_state, int):
            np.random.seed(random_state)
        prop = np.random.dirichlet(np.ones(num_celltype), samplenum) 

    elif d_prior is not None:
        print('Using prior info to generate cell fractions in Dirichlet distribution')
        assert len(d_prior) == num_celltype, 'dirichlet prior is a vector, its length should equals to the number of cell types'
        if isinstance(random_state, int):
            np.random.seed(random_state)
        prop = np.random.dirichlet(d_prior, samplenum)
        print('Dirichlet cell fractions is generated')

    # make the dictionary
    for key, value in celltype_groups.items():
        celltype_groups[key] = np.array(value)
    prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

    # sparse cell fractions
    if sparse:
        print("You set sparse as True, some cell's fraction will be zero, the probability is", sparse_prob)
        ## Only partial simulated data is composed of sparse celltype distribution
        for i in range(int(prop.shape[0] * sparse_prob)): 
            indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * sparse_prob))
            prop[i, indices] = 0

        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

    if rare:
        print(
            'You will set some cell type fractions are very small (<3%), '
            'these celltype is randomly chosen by percentage you set before.')
        ## choose celltype
        np.random.seed(0)
        indices = np.random.choice(np.arange(prop.shape[1]), replace=False, size=int(prop.shape[1] * rare_percentage))
        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

        for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
            prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
            buf = prop[i, indices].copy()
            prop[i, indices] = 0
            prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
            prop[i, indices] = buf

    # precise number for each celltype
    cell_num = np.floor(n * prop)

    # precise proportion based on cell_num
    prop = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)

    # start sampling
    sample = np.zeros((prop.shape[0], sc_data.shape[1]))
    allcellname = celltype_groups.keys()
    print('Sampling cells to compose pseudo-bulk data')
    for i, sample_prop in tqdm(enumerate(cell_num)):
        for j, cellname in enumerate(allcellname):
            select_index = choice(celltype_groups[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data[select_index].sum(axis=0)

    prop = pd.DataFrame(prop, columns=celltype_groups.keys())
    simudata = anndata.AnnData(X=sample,
                               obs=prop,
                               var=pd.DataFrame(index=genename))

    labelname = celltype_groups.keys()
    print('Sampling is done')

    return simudata,labelname

def generate_simulated_dataVarTiss(sc_data,
                                   tissue_list=None,  d_prior=None,
                                   n=500, samplenum=5000,
                                   random_state=None, sparse=True, sparse_prob=0.5):

    print('Reading single-cell dataset, this may take 1 min')
    if '.txt' in sc_data:
        sc_data = pd.read_csv(sc_data, index_col=0, sep='\t')
        sc_data.dropna(inplace=True)
        sc_data['celltype'] = sc_data.index
        print(sc_data.index.unique())
        sc_data.index = range(len(sc_data))

    if tissue_list is not None:
        print(f"Filtering dataset to only include tissues: {tissue_list}")
        sc_data = sc_data[sc_data['celltype'].isin(tissue_list)]

    num_celltype = len(sc_data['celltype'].value_counts())
    genename = sc_data.columns[:-1]

    sc_data2 = sc_data.copy()
    sc_data2.reset_index(drop=True, inplace=True)
    celltype_groups = sc_data2.groupby('celltype').groups
    sc_data2.drop(columns='celltype', inplace=True)
    sc_data2 = anndata.AnnData(sc_data2)
    sc_data2 = sc_data2.X
    sc_data2 = np.ascontiguousarray(sc_data2, dtype=np.float32)

    print('Using prior info to generate cell fractions in Dirichlet distribution')
    assert len(d_prior) == num_celltype, 'dirichlet prior is a vector, its length should equals to the number of cell types'
    if isinstance(random_state, int):
        np.random.seed(random_state)
    prop = np.random.dirichlet(d_prior, samplenum)
    print('Dirichlet cell fractions is generated')
    prop = prop / np.sum(prop, axis=1, keepdims=True)

    if sparse:
        print(f"Sparse mode on, probability = {sparse_prob}")
        for i in range(int(prop.shape[0] * sparse_prob)):
            indices = np.random.choice(np.arange(prop.shape[1]), 
                                        replace=False, 
                                        size=int(prop.shape[1] * sparse_prob))
            prop[i, indices] = 0
        prop = prop / np.sum(prop, axis=1, keepdims=True)

    # precise number for each celltype
    cell_num = np.floor(n * prop)
    prop = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)

    cell_num_sample0 = cell_num[0]
    tissue_list = list(celltype_groups.keys())
    plt.figure(figsize=(6, 4))
    plt.boxplot(cell_num, vert=True, patch_artist=True, 
            boxprops=dict(facecolor='#ff9999', color='#ff9999'),
            medianprops=dict(color='red'))
    plt.ylabel("Tissue sample number")
    plt.xlabel("Tissue type")
    plt.title("Distribution of tissue sample number for simulation")
    plt.xticks(np.arange(1, len(tissue_list) + 1), tissue_list)
    plt.show()

    sample = np.zeros((prop.shape[0], sc_data2.shape[1]))
    allcellname = list(celltype_groups.keys())
    print('Sampling cells to compose pseudo-bulk data')
    for i, sample_prop in tqdm(enumerate(cell_num), total=len(cell_num)): 
        for j, cellname in enumerate(allcellname):
            # print(cellname)
            select_index = choice(celltype_groups[cellname], size=int(sample_prop[j]), replace=True) 
            sample[i] += sc_data2[select_index].sum(axis=0)

    prop = pd.DataFrame(prop, columns=allcellname)
    simudata = anndata.AnnData(X=sample,
                                obs=prop,
                                var=pd.DataFrame(index=genename))
    print('Sampling is done')

    return simudata, allcellname, cell_num, prop

def GTEDataGeneVar(train_data, vart = True, variance_threshold=0.98, sample_porp=[0.3, 0.6, 0.9]):
    train_x = pd.DataFrame(train_data.X, columns=train_data.var.index)

    if vart:
        ### variance cutoff
        print('Cutting variance...')
        var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[int(train_x.shape[1] * variance_threshold)]
        train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]
        var_genename = list(train_x.columns)

        # Sample genes based on the specified sample sizes
        num_genes = train_x.shape[1]
        sample_sizes = np.round([sample_porp[0] * num_genes, sample_porp[1] * num_genes, sample_porp[2] * num_genes]).astype(int)

        sampled_genes = {}
        for size in sample_sizes:
            if size <= len(var_genename):
                sampled_genes[size] = np.random.choice(var_genename, size=size, replace=False)
            else:
                print(f"Warning: Requested sample size {size} exceeds the available gene list size. Returning all genes.")
                sampled_genes[size] = var_genename

        return var_genename, sampled_genes

    else:
       return list(train_x.columns)

def GTEDataGeneVarIntv(train_data):

    train_x = pd.DataFrame(train_data.X, columns=train_data.var.index)
    vars = train_x.var(axis=0).sort_values(ascending=False)
    var_cutoff1 = vars[int(train_x.shape[1] * 0.3)] 
    var_cutoff2 = vars[int(train_x.shape[1] * 0.6)] 
    var_cutoff3 = vars[int(train_x.shape[1] * 0.9)]

    train_x1 = train_x.loc[:, vars > var_cutoff1]
    var_genename1 = list(train_x1.columns)

    train_x2 = train_x.loc[:, (vars > var_cutoff2) & (vars <= var_cutoff1)]
    var_genename2 = list(train_x2.columns)

    train_x3 = train_x.loc[:, (vars > var_cutoff3) & (vars <= var_cutoff2)]
    var_genename3 = list(train_x3.columns)

    sampled_genes = {
        'high_variance': var_genename1,
        'medium_variance': var_genename2,
        'low_variance': var_genename3
    }
    return sampled_genes
 
def GTEDataGeneVarReal(train_data, variance_threshold=0.98, sample_porp=[0.3, 0.6, 0.9]):

    train_x = pd.read_csv(train_data, index_col=0, sep='\t')

    ### variance cutoff
    print('Cutting variance...')
    var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[int(train_x.shape[1] * variance_threshold)]
    train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]

    ### find intersected genes
    print('Finding intersected genes...')
    var_genename = list(train_x.columns)

    # Sample genes based on the specified sample sizes
    num_genes = train_x.shape[1]
    sample_sizes = np.round([sample_porp[0] * num_genes, sample_porp[1] * num_genes, sample_porp[2] * num_genes]).astype(int)

    sampled_genes = {}
    for size in sample_sizes:
        if size <= len(var_genename):
            sampled_genes[size] = np.random.choice(var_genename, size=size, replace=False)
        else:
            print(f"Warning: Requested sample size {size} exceeds the available gene list size. Returning all genes.")
            sampled_genes[size] = var_genename

    return var_genename, sampled_genes
  
def GetXandYSelG(train_data, sel_genes, scaler="mms"):
    train_x = pd.DataFrame(train_data.X, columns=train_data.var.index)
    train_y = train_data.obs
    celltypes = train_y.columns
    train_x = train_x[train_x.columns.intersection(sel_genes)]

    # MinMax process
    print('Scaling...')
    train_x = np.log(train_x + 1)

    if scaler=='ss':
        print("Using standard scaler...")
        ss = StandardScaler()
        ss_train_x = ss.fit_transform(train_x.T).T
        return ss_train_x, train_y.values, celltypes

    elif scaler == 'mms':
        print("Using minmax scaler...")
        mms = MinMaxScaler()
        mms_train_x = mms.fit_transform(train_x.T).T
        return mms_train_x, train_y.values, celltypes

def GetXandYSelGReal(train_data, sel_genes, scaler="mms"):

    train_x = pd.read_csv(train_data, index_col=0, sep='\t')
    train_x = train_x[train_x.columns.intersection(sel_genes)]

    # MinMax process
    print('Scaling...')
    train_x = np.log(train_x + 1)

    if scaler=='ss':
        print("Using standard scaler...")
        ss = StandardScaler()
        ss_train_x = ss.fit_transform(train_x.T).T
        return ss_train_x

    elif scaler == 'mms':
        print("Using minmax scaler...")
        mms = MinMaxScaler()
        mms_train_x = mms.fit_transform(train_x.T).T
        return mms_train_x

def PlotData(GTE_dat, HPA_dat, real_dat, per = 0.3):
        
    fig = plt.figure()
    sns.histplot(data=np.mean(GTE_dat, axis=0), kde=True, color='#f46d43',edgecolor=None)
    sns.histplot(data=np.mean(HPA_dat, axis=0), kde=True, color='#fee08b',edgecolor=None)
    sns.histplot(data=np.mean(real_dat, axis=0), kde=True, color='#66bd63',edgecolor=None)
    plt.legend(title='Percent' + str(per), labels=["GTE", "HPA", "real"])
    plt.show()

def PlotDataTwoSet(GTE_dat, HPA_dat, per = 0.3, labs = ["GTE", "HPA"]):
    fig = plt.figure()
    sns.histplot(data=np.mean(GTE_dat, axis=0), kde=True, color='#f46d43',edgecolor=None)
    sns.histplot(data=np.mean(HPA_dat, axis=0), kde=True, color='#fee08b',edgecolor=None)
    plt.legend(title='Percent' + str(per), labels=labs)
    plt.show()

def PlotDataTsne(GTE_dat, HPA_dat, real_dat, per = 0.3):
    all_data = np.concatenate([GTE_dat, HPA_dat, real_dat], axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_data)
    labels = ['GTE'] * GTE_dat.shape[0] + ['HPA'] * HPA_dat.shape[0] + ['real'] * real_dat.shape[0]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette=['#f46d43', '#fee08b', '#66bd63'], s=50, alpha=0.7)
    plt.title(f't-SNE plot (Percent = {per})', fontsize=16)
    plt.legend(title='Dataset', labels=["GTE", "HPA", "real"])
    plt.show()

def PlotDataTsneTwoSet(GTE_dat, HPA_dat, per = 0.3, labs = ["GTE", "HPA"]):
    all_data = np.concatenate([GTE_dat, HPA_dat], axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_data)
    labels = ['GTE'] * GTE_dat.shape[0] + ['HPA'] * HPA_dat.shape[0]
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette=['#f46d43', '#fee08b'], s=50, alpha=0.7)
    plt.title(f't-SNE plot (Percent = {per})', fontsize=16)
    plt.legend(title='Dataset', labels=labs)
    plt.show()

def DataSplitTrValTe(sim_scaled, simulated_y, test_size=0.2, val_size=0.2):

    indices = np.arange(sim_scaled.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=2025)
    simulated_x_train = sim_scaled[train_indices]
    simulated_x_test = sim_scaled[test_indices]
    simulated_y_train = simulated_y[train_indices]
    simulated_y_test = simulated_y[test_indices]
    # simulated_y_train = simulated_y.iloc[train_indices]
    # simulated_y_test = simulated_y.iloc[test_indices]

    if val_size is not None:
        relative_val_size = val_size / (1 - test_size)
        train_indices, val_indices = train_test_split(train_indices, test_size=relative_val_size, random_state=2025)
        simulated_x_val = sim_scaled[val_indices]
        simulated_y_val = simulated_y[val_indices]

    else:
        simulated_x_val = None
        simulated_y_val = None

    return simulated_x_train, simulated_x_val, simulated_x_test, simulated_y_train, simulated_y_val, simulated_y_test

