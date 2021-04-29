# -*- coding: utf-8 -*-
"""
Change point analysis with Ruptures and Bayesian detection.

Structure: 
    Imports
    Generate data
    Ruptures detection
    Bayesian detection
    Plot of results
    Error estimation

@author: DEREGED1
"""

# %% Imports
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

from functools import partial

# Detection
import ruptures as rpt
import bayesian_changepoint_detection.offline_changepoint_detection as offcd

# %% Generate data

def generate_normal_time_series(num, minl=50, maxl=1000):
    index_count = 0
    BP = []
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.randn()*10
        var = np.random.randn()*1
        if var < 0:
            var = var * -1
        tdata = np.random.normal(mean, var, p)
        data = np.concatenate((data, tdata))
        index_count += len(tdata)
        BP.append(index_count)
    return data, (BP)

data , CPs_true = generate_normal_time_series(7, 50, 200)

# %% Ruptures detection

## When n_bkps unknown
# PELT
algo = rpt.Pelt(model = 'linear').fit(data)

# Binary segmentation
#algo = rpt.Binseg().fit(data)

# Bottum Up
#algo = rpt.BottomUp().fit(signal)

# Kernel
#algo = rpt.KernelCPD().fit(signal)

# Window
#algo = rpt.Window().fit(signal)

bkps_RUPTURES = algo.predict(pen = 200)

## When n_bkps known
# Dynamic programming
#algo = rpt.Dynp().fit(signal)

#bkps_RUPTURES = algo.predict(n_bkps = 4)

predicted_cp_RUPTURES = np.zeros(data.shape)
predicted_cp_RUPTURES[np.array(bkps_RUPTURES)-1]=1

# Display
rpt.display(data, CPs_true, bkps_RUPTURES)
plt.show()

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
predicted_cp_BAYES = np.exp(Pcp).sum(0)
bkps_BAYES = np.where(predicted_cp_BAYES > 0.7)[0]

# Display
rpt.display(data, bkps, bkps_BAYES)
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
ax.plot(predicted_cp_BAYES)
ax.set_title('Bayesian', fontsize=14)

# %% Plot together
plt.figure(figsize=(20,6))
plt.plot(data[:])
plt.scatter(np.array(bkps)-1, data[np.array(bkps)-1], linewidths=5)
plt.scatter(np.array(bkps_RUPTURES)-1, data[np.array(bkps_RUPTURES)-1], linewidths=5)
plt.scatter(np.array(bkps_BAYES)-1, data[np.array(bkps_BAYES)-1], linewidths=5)
plt.legend(['Data','True','Ruptures','Bayes'])

