import os
import anndata
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import pearsonr

import torch
import random
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def counts2FPKM(counts, genelen):
    genelen = pd.read_csv(genelen, sep=',')
    genelen['TranscriptLength'] = genelen['Transcript end (bp)'] - genelen['Transcript start (bp)']
    genelen = genelen[['Gene name', 'TranscriptLength']]
    genelen = genelen.groupby('Gene name').max()
    # intersection
    inter = counts.columns.intersection(genelen.index)
    samplename = counts.index
    counts = counts[inter].values
    genelen = genelen.loc[inter].T.values
    # transformation
    totalreads = counts.sum(axis=1)
    counts = counts * 1e9 / (genelen * totalreads.reshape(-1, 1))
    counts = pd.DataFrame(counts, columns=inter, index=samplename)
    return counts

def FPKM2TPM(fpkm):
    genename = fpkm.columns
    samplename = fpkm.index
    fpkm = fpkm.values
    total = fpkm.sum(axis=1).reshape(-1, 1)
    fpkm = fpkm * 1e6 / total
    fpkm = pd.DataFrame(fpkm, columns=genename, index=samplename)
    return fpkm

def counts2TPM(counts, genelen):
    fpkm = counts2FPKM(counts, genelen)
    tpm = FPKM2TPM(fpkm)
    return tpm

def L1error(pred, true):
    return np.mean(np.abs(pred - true))

def CCCscore(y_pred, y_true, mode='all'):
    # pred: shape{n sample, m cell}
    if mode == 'all':
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
    elif mode == 'avg':
        pass
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_value += ccc
    return ccc_value / y_pred.shape[1]

def score(pred, label):
    print('L1 error is', L1error(pred, label))
    print('CCC is ', CCCscore(pred, label))

def showloss(loss):
    sns.set()
    plt.plot(loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

def transformation(train_x, test_x):
    sigma_2 = np.sum((train_x - np.mean(train_x, axis=0)) ** 2, axis=0) / (train_x.shape[0] + 1)
    sigma = np.sqrt(sigma_2)
    test_x = ((test_x - np.mean(test_x, axis=0)) / np.std(test_x, axis=0)) * sigma + np.mean(train_x, axis=0)
    return test_x

class simdatset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y

class realdatset(Dataset):
    def __init__(self, X):
        self.X = X


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)

        return x
    
def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


# 行是样本，列是基因/占比
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def calculate_evaluation_metrics(x_recon_res, input_X, out_pth=None):
    # Convert to numpy arrays if they are DataFrames
    if isinstance(x_recon_res, pd.DataFrame):
        x_recon_res = x_recon_res.to_numpy()
    if isinstance(input_X, pd.DataFrame):
        input_X = input_X.to_numpy()
    
    # Initialize lists to store metrics for each sample
    rmse_m = []
    pearson_corr_value = []
    mae_m = []
    ccc_values = []
    
    # Calculate metrics for each sample
    for i in range(input_X.shape[0]):
        # RMSE
        rmse = np.sqrt(mean_squared_error(input_X[i, :], x_recon_res[i, :]))
        rmse_m.append(rmse)
        
        # Pearson Correlation
        corr, _ = pearsonr(input_X[i, :], x_recon_res[i, :])
        if not np.isnan(corr):
            pearson_corr_value.append(corr)
        
        # MAE
        mae = mean_absolute_error(input_X[i, :], x_recon_res[i, :])
        mae_m.append(mae)
        
        # Lin's CCC (using sklearn's variance and covariance)
        mean_x = np.mean(input_X[i, :])
        mean_y = np.mean(x_recon_res[i, :])
        var_x = np.var(input_X[i, :], ddof=0)  # ddof=0 for population variance
        var_y = np.var(x_recon_res[i, :], ddof=0)
        cov_xy = np.cov(input_X[i, :], x_recon_res[i, :], ddof=0)[0, 1]
        ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)**2)
        ccc_values.append(ccc)
    
    # Calculate average metrics
    average_rmse = np.mean(rmse_m)
    average_pearson_corr = np.mean(pearson_corr_value)
    average_mae = np.mean(mae_m)
    average_ccc = np.mean(ccc_values)
    
    # Save metrics to CSV files
    if out_pth is not None:
        pd.DataFrame(rmse_m, columns=["RMSE"]).to_csv(out_pth + "rmse_values.csv", index=False)
        pd.DataFrame(pearson_corr_value, columns=["PCC"]).to_csv(out_pth + "pearson_corr_values.csv", index=False)
        pd.DataFrame(mae_m, columns=["MAE"]).to_csv(out_pth + "mae_values.csv", index=False)
        pd.DataFrame(ccc_values, columns=["CCC"]).to_csv(out_pth + "ccc_values.csv", index=False)

    return rmse_m, average_rmse, pearson_corr_value, average_pearson_corr, mae_m, average_mae, ccc_values, average_ccc