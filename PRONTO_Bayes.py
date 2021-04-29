# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:01:10 2021

A file to calculate Bayesian posterior distribution used for predicting 
change points in data. 

@author: DEREGED1
"""

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
import bayesian_changepoint_detection.offline_changepoint_detection as offcd

# %% COLLECT DATA

# EXPERIMENT DAY 4: 12-09-2017
df = pd.read_csv('PRONTO_dataset/TD_0912/0912Testday4.csv')

# %% Preparation of df

df.columns =  df.iloc[1].values
df = df.drop(0, axis = 0)
df = df.drop(1, axis = 0)
df = df.reset_index(drop = True)


# %% Select relevant columns
df = df[['TIMESTAMP','Air In1', 'Air In2', 'Water In1','Water In2']]
df = df.loc[:, ~df.columns.duplicated()]
# %% Consvert CPs timestamps to index

CPS_0912_timestamps = ['10:35', '10:43', '10:50', '10:58', '11:08', '11:19', 
                       '11:33', '11:51', '12:13', '12:30', '12:38', '12:49', 
                       '12:58', '13:06', '13:18', '13:27']
CPS_0911_timestamps = ['10:40', '10:50', '11:03', '11:28', '11:30', '11:55', 
                       '12:03', '12:13', '12:23', '12:33', '12:37', '12:38', 
                       '12:46', '13:10', '13:18', '13:26', '13:33', '13:44', 
                       '14:49', '14:56', '15:02', '15:10', '15:20', '15:30', 
                       '15:37', '15:46', '15:52', '15:59']
CPS_0907_timestamps = ['13:01', '13:26', '13:38', '13:44', '13:52', '14:00', 
                       '14:08', '14:17', '14:26', '14:34', '14:35', '14:50',
                       '14:58', '15:06', '15:14', '15:22', '15:30', '15:38',
                       '15:46', '15:54', '16:00', '16:10', '16:15', '16:22', 
                       '16:28', '16:33', '16:38']

CPS_0912 = []

for i in CPS_0912_timestamps:
    df_temp = df[df['TIMESTAMP'] == '09/12/2017 '+i]
    #print(np.mean(df_temp.index))
    CPS_0912.append(int(np.floor(np.mean(df_temp.index))))

CPS_0912.append(df.shape[0])
df = df.drop('TIMESTAMP', axis = 1)

# %%
start = 2000
end = 3100
for col in df.columns:
    data = np.stack(df[col].values[start:end]).astype(None)
    
    # Chose prior (constant, geometric, negBin)
    prior = partial(offcd.const_prior, l=(len(data)+1))
    #prior = partial(offcd.const_prior, l=2000)
    #prior = partial(offcd.geometric_prior, p = (1/100000))
    #prior = partial(offcd.neg_binominal_prior, k = 5, p = 0.00001)

    Q, P, Pcp = offcd.offline_changepoint_detection(np.array(data), 
                                                prior, 
                                                offcd.gaussian_obs_log_likelihood, 
                                                truncate=-40)

    # Get most likely indexes
    predicted_cp_BAYES = np.exp(Pcp).sum(0)
    predicted_cp_BAYES = pd.DataFrame(predicted_cp_BAYES)
    predicted_cp_BAYES.to_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_'+col+'.csv')
    print('Done with '+ col)
    
    # %%
    
pcp_Air1 = pd.read_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_Air In1.csv')
pcp_Air2 = pd.read_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_Air In2.csv')
pcp_Water1 = pd.read_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_Water In1.csv')
pcp_Water2 = pd.read_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_Water In2.csv')
   
plt.plot(pcp_Air1['0'])
plt.plot(pcp_Air2['0'])
plt.plot(pcp_Water1['0'])
plt.plot(pcp_Water2['0'])


