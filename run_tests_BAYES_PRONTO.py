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
from sklearn.preprocessing import MinMaxScaler

# %% COLLECT DATA

# EXPERIMENT DAY 4: 12-09-2017
df = pd.read_csv('TD_0912/0912Testday4.csv')

# %% Preparation of df
df.columns =  df.iloc[1].values
df = df.drop(0, axis = 0)
df = df.drop(1, axis = 0)
df = df.reset_index(drop = True)


# %% Select relevant columns
df = df[['TIMESTAMP','Air In1', 'Air In2', 'Water In1','Water In2']]
df = df.loc[:, ~df.columns.duplicated()]
# %% Consvert CPs timestamps to index

# Normalise process varaibles
# NORMALISED SIGNALS
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df.values)
norm_df = pd.DataFrame(x_scaled)
norm_df.columns = df.columns

# Apply Piecewise Aggregate Approximation


# %%
# Calculate posterior distributions on PAA data
for col in df.columns:
    data = np.stack(df[col].values).astype(None)
    
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
# To read already calculated posterior distributions
pcp_Air1 = pd.read_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_Air In1.csv')
pcp_Air2 = pd.read_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_Air In2.csv')
pcp_Water1 = pd.read_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_Water In1.csv')
pcp_Water2 = pd.read_csv('PRONTO_dataset/TD_0912/BAYESIAN_pcp_Water In2.csv')

# Plot the posterior distributions
plt.plot(pcp_Air1['0'])
plt.plot(pcp_Air2['0'])
plt.plot(pcp_Water1['0'])
plt.plot(pcp_Water2['0'])


