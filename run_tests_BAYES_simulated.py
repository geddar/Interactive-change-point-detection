# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:01:10 2021

Change point analysis with Bayesian detection.
A file to calculate Bayesian posterior distribution used for predicting 
change points in simulated datasets. 

Structure: 
    Imports
    Generate data
    Ruptures detection
    Bayesian detection
    Plot of results
    Error estimation

@author: Rebecca Gedda
"""

# %% Imports
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

from functools import partial

# Detection
import ruptures as rpt
import bayesian_changepoint_detection.offline_changepoint_detection as offcd

# %% Import data


# %% Bayesian detection

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
Bayes_pcp = np.exp(Pcp).sum(0)


cp_bayes, _ = find_peaks(Bayes_pcp.values.reshape(1,-1)[0], height = 0.5, distance = 10)
cp_bayes = np.concatenate([cp_bayes, [data.shape[0]]])

CPs_bayes = pd.DataFrame(cp_bayes)
CPs_bayes.to_csv('Simulated data/'+problem_tag+'/CPs_bayes.dat')


# Display
rpt.display(data, bkps, CPs_bayes)
plt.show()

# %% Plot of results

fig, ax = plt.subplots(figsize=[18, 16])

ax = fig.add_subplot(3, 1, 1)
ax.plot(data[:])
#ax.plot(np.array(bkps)-1, data[np.array(bkps)-1], 'o')
#ax.plot(predicted_cp, data[predicted_cp])

ax = fig.add_subplot(3, 1, 2, sharex=ax)
#ax.plot(predicted_cp_RUPTURES)
ax.set_title('Ruptures', fontsize=14)


ax = fig.add_subplot(3, 1, 3, sharex=ax)
ax.plot(Bayes_pcp)
ax.set_title('Bayesian', fontsize=14)

# %% Plot together
plt.figure(figsize=(20,6))
plt.plot(data[:])
plt.scatter(np.array(bkps)-1, data[np.array(bkps)-1], linewidths=5)
plt.scatter(np.array(bkps_RUPTURES)-1, data[np.array(bkps_RUPTURES)-1], linewidths=5)
plt.scatter(np.array(bkps_BAYES)-1, data[np.array(bkps_BAYES)-1], linewidths=5)
plt.legend(['Data','True','Ruptures','Bayes'])

