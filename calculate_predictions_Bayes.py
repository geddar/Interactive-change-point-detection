# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:09:50 2021

IN: Dataset and CPs
OUT: Created files with predictions in Results folder

Creates a file for each calculated posterior distribution
and a file for predicted change points. 

@author: Rebecca Gedda
"""

##############################################################################
#                    PREDICTIONS WITH BAYES
##############################################################################

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ruptures import metrics as rpt_met
import os

from pyts.approximation import PiecewiseAggregateApproximation as PAA
from scipy.signal import find_peaks

from functools import partial
import offline_changepoint_detection as offcd

### FUNCTIONS ################################################################

def select_prior(name):
    if name == 'flat':
        prior = partial(offcd.const_prior, l=(data.shape[0]+1))
        #prior = partial(offcd.const_prior, l=2000)
    elif name == 'geometric':
        prior = partial(offcd.geometric_prior, p = (1/100000))
    elif name == 'neg_binomial':
        prior = partial(offcd.neg_binominal_prior, k = 5, p = 0.00001)
    else:
        print('Select a prior from: flat, geometric, neg_binomial')
    return prior

def select_likelihood(name):
    if name == 'gaussian':
        likelihood = offcd.gaussian_obs_log_likelihood
    elif name == 'imf':
        likelihood =  offcd.ifm_obs_log_likelihood          
    elif name == 'fullcov':
        likelihood = offcd.fullcov_obs_log_likelihood
    else:
        print('Select a likelihood from: gaussian, imf, fullcov')
    return likelihood               
           
def print_metrics(tag, data, CPs_true, CPs, PAA_size, prior_name, likelihood_name):
    
    # Metrics
    K = len(CPs)
    
    # Annotation error
    AE = np.abs(K - len(CPs_true))
    
    # Meantime
    MT = rpt_met.meantime(CPs_true, CPs)
    
    # RandIndex
    RI = rpt_met.randindex(CPs, CPs_true)
    
    # Precision/Recall (Note margin can be changed)
    margin = np.ceil(data.shape[0]*0.01)
    #margin = 10
    Precision = rpt_met.precision_recall(CPs_true, CPs, margin = margin)
    
    # F1-score
    if Precision[0] + Precision[1] != 0:
        F1 = (2*Precision[0]*Precision[1])/(Precision[0] + Precision[1])
    else:
        F1 = 0
    
    print('----- Results for '+ tag +' with Bayesian approach'
                  ' ----- \n Using: '+ prior_name +'prior and '+ likelihood_name +
                  ' likelihood function \n PAA window: '+ str(PAA_size)+
                  '\n \n' +  str(CPs) +
                  '\n K: ' + str(K) +
                  '\n AE: '+ str(AE) +
                  '\n MT: ' + str(MT) +
                  '\n Precision: '+ str(Precision[0])+
                  '\n Recall: ' +str(Precision[1])+
                  '\n F1: ' + str(F1) +
                  '\n RI: ' + str( RI) +
                  '\n ------------------------------------------- \n \n')

def read_dataset_folder(name_tag):
    # Read data and true change points
    directory = 'Datasets/'+name_tag
    
    for filename in os.listdir(directory):
        if filename == 'CPs.dat':
            CPs = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
            CPs = CPs['0'].values
        elif filename!='Results':
            #data = pd.read_csv(filename)
            data = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
    del directory, filename
    return data, CPs

def plot_shifting_backgorund(data, cps):
    plt.figure(figsize=(16,4))
    plt.plot(data, label = 'Data')
    start = 0
    colour = 0.3
    for end in cps:
        plt.axvspan(start, end, facecolor='grey', alpha=colour)
        start = end
        
        if colour == 0.3: 
            colour = 0.2
        else:
            colour = 0.3
            
### MAIN SCRIPT ##############################################################

data_tag = 'P4'                     # P1, P2, P3, P4, P5, P6, PRONTO

# Settings                          # SELECTIONS
prior_function = 'flat'             # flat, geometric, neg_binomial
likelihood_function = 'gaussian'    # gaussian, imf, fullcov

confidence_level = 0.5              # Double, [0,1]
distance_between_CPs = 10           # Int, >= 0

PAA_window_size = 2                # Integer. If == 1, no PAA is applied

print_results = True               # True or False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CALCULATE POSTERIOR DISTRIBUTION

# read data and truechange points
data, CPs_true = read_dataset_folder(data_tag)

# Prior selection
prior = select_prior(prior_function)

# Likelihood selection
likelihood = select_likelihood(likelihood_function)

# If there exists a calculated prior
directory = 'Datasets/'+ data_tag +'/Results'
if 'pcp_'+ prior_function +'_PAA_'+ str(PAA_window_size) +'.dat' in os.listdir(directory):
    pcp = pd.read_csv(directory + '/pcp_'+ prior_function +'_PAA_'+ str(PAA_window_size) +'.dat')
    pcp = pcp['0']

# Otherwise, calculate the prior and save
else:
    # Set parameters for PAA
    paa = PAA( window_size= PAA_window_size)
    
    pcps = np.zeros( int(np.ceil(data.shape[0]/PAA_window_size)) -1)
    for dim in range(data.shape[1]):
        
        sub_data = paa.transform( np.array(data[data.columns[dim]] ).reshape(1, -1) )
        
        Q, P, Pcp = offcd.offline_changepoint_detection(sub_data[0], 
                                                    prior, 
                                                    likelihood, 
                                                    truncate=-40)
    
        # Get most likely indexes
        sub_pcp = np.exp(Pcp).sum(0)
        pcps += sub_pcp
    # Aggregate over dims
    pcp = pd.DataFrame(pcps)
    
    # Save combined posterior distribution
    pcp.to_csv( directory+'/pcp_'+ prior_function +'_PAA_'+ str(PAA_window_size) +'.dat')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IDENTIFY CHANGE POINTS

cp_bayes, _ = find_peaks(pcp.values.reshape(1,-1)[0], height = confidence_level, 
                         distance = distance_between_CPs)
# Adjust for PAA and add last index
cp_bayes = cp_bayes*PAA_window_size + np.floor(PAA_window_size/2)
cp_bayes = np.concatenate([cp_bayes.astype(int), [data.shape[0]]])

# Turn into dataframe and save predictions
CPs_bayes = pd.DataFrame(cp_bayes)
CPs_bayes.to_csv(directory+'/Bayes_CPs_'+ prior_function +'_PAA_'+ str(PAA_window_size) + '_conf_'+ str(confidence_level) +'.dat')

if print_results:
    print_metrics(data_tag, data, CPs_true, cp_bayes, PAA_window_size, prior_function, likelihood_function)
else:
    print('----- Done with '+ data_tag +' with Bayesian approach'
                  ' ----- \n Using: '+ prior_function +' prior and '+ likelihood_function +
                  ' likelihood function \n PAA window: '+ str(PAA_window_size)+
                  '\n ------------------------------------------- \n \n')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PLOTS

if data.shape[1]!=1:
    data = data[data.columns[0]]

plot_shifting_backgorund(data, CPs_true)
plt.plot(CPs_bayes.values, np.ones(CPs_bayes.shape[0])*-0.1, 'X', label = 'Bayes')
plt.title('Predictions on ' + data_tag + ' using Bayesian approach ('+ prior_function +' prior, '+ likelihood_function +' likelihood)', fontsize = 14)
plt.legend(loc = 'upper left', fontsize = 14)