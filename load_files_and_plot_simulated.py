# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:58:50 2021

A script to explore datafiles (not genreate, load from directory).
uses the file structure found in this repository.

@author: Rebecca Gedda
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import ruptures as rpt
#from scipy.signal import find_peaks

#  FUNCTIONS 
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def read_directory_files(path):
    import os

    directory = path
    
    df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            dfn = pd.read_csv(os.path.join(directory, filename))
            #print(dfn.head())
            df = pd.concat([df, dfn], ignore_index=True)
        elif filename == 'CPs.dat':
            CPs = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
        elif filename == 'Bayes_flat_prior.dat':
            Bayes_pcp = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
        elif filename == 'CPs_bayes.dat':
            CPs_bayes = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
        else:
            #data = pd.read_csv(filename)
            data = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
    return data, CPs, Bayes_pcp, CPs_bayes

def read_results(path):
    import os

    directory = path
    
    df = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            dfn = pd.read_csv(os.path.join(directory, filename))
            #print(dfn.head())
            df = pd.concat([df, dfn], ignore_index=True)
    return df

# Function to plot semgents created by change points in different colours.
def plot_shifting_backgorund(data, cps):
    plt.figure(figsize=(16,3))
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
        
def print_metrics(CPs_true, CPs_predictions):
    import ruptures.metrics as rpt_met
    
    print('Number of predictions (K): '+ str(CPs_predictions.shape[0]))
    
    # Annotation error
    AE = np.abs(CPs_true.shape[0] - CPs_predictions.shape[0])
    print('AE: '+ str(AE))

    # Meantime
    MT = rpt_met.meantime(CPs_true.values.reshape(1,-1)[0], CPs_predictions.values.reshape(1,-1)[0])
    print('Meanitme: '+ str(MT))
    
    # Precision/Recall (Note margin can be changed)
    #margin = 10
    margin = np.ceil(CPs_true.values[-1]*0.05)
    Precision = rpt_met.precision_recall(CPs_true.values.reshape(1,-1)[0], CPs_predictions.values.reshape(1,-1)[0], margin = margin)
    print('Precision: '+ str(Precision[0]))
    print('Precision: '+ str(Precision[1]))
    
    # F1-score
    #F1 = 2* (Perc * recall)/(Perc + recall)
    F1 = (2*Precision[0]*Precision[1])/(Precision[0] + Precision[1])
    print('F1: '+ str(F1))
    
    # RandIndex
    RI = rpt_met.randindex(CPs_predictions.values.reshape(1,-1)[0], CPs_true.values.reshape(1,-1)[0])
    print('RandIndex: '+ str(RI))


# %% Read all files in a directory
problem_tag = 'P6'

data, CPs_true, Bayes_pcp, CPs_bayes = read_directory_files('Simulated data/'+problem_tag)

df = read_results('Result metrics/'+ problem_tag)

# Divide into PELT and WIN
df_PELT = df[df['Search_direction']== 'PELT']
df_WIN = df[df['Search_direction']=='WIN']

del df


# %% PRINT METRICS FOR SOME PREDICTIONS
print('BAYESIAN PREDICTIONS')
print(problem_tag)
print_metrics(CPs_true, CPs_bayes)

# %% Plot Bayesian PCP

fig, axs = plt.subplots(2, 1, sharex=(True), figsize = [18,6])

axs[0].plot(data)
axs[0].plot(CPs_true.values.reshape(1,-1)[0] - 1, data.values.reshape(1,-1)[0][CPs_true.values.reshape(1,-1)[0] - 1], 'o')
axs[0].plot(CPs_bayes.values.reshape(1,-1)[0] - 2, data.values.reshape(1,-1)[0][CPs_bayes.values.reshape(1,-1)[0] - 2], 'X', color = 'black')
axs[0].set_title('Exponential decay data', fontsize=20)

axs[1].plot(Bayes_pcp)
axs[1].plot(CPs_bayes.values.reshape(1,-1)[0][:-1] - 1, Bayes_pcp.values.reshape(1,-1)[0][CPs_bayes.values.reshape(1,-1)[0][:-1] ], 'X', color = 'black')
axs[1].set_title('Bayesian Posterior distribution', fontsize=16)

fig.tight_layout()
plt.show()


# %% Plot data and true datapoints

# Select some change points to plot
cp1 = [ 55, 200, 375, 550, 700]
cp2 = [ 50, 155, 330, 505, 700]
cp3 = [ 50, 155, 330, 505, 700]

plot_shifting_backgorund(data, CPs_true.values)
plt.title('Piecewise linear (WIN)', fontsize=16)

plt.plot(cp2, np.ones(len(cp2))*-5,  'o', label = "AR")
plt.plot(cp1, np.ones(len(cp1))*-5.6,  'o', label = "L1")
plt.plot( cp3, np.ones(len(cp3))*-6,  'o', label = "L2")

plt.plot( CPs_bayes, np.ones(len(CPs_bayes))*-7, 'X', label = 'Bayesian' )
plt.legend(loc='upper left', fontsize=16)

