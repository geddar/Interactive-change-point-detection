# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:10:31 2021

Script to visualise the results in a dataset folder.

@author: Rebecca Gedda
"""

##############################################################################
#                    VISUALISE PREDICTIONS
##############################################################################

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

### FUNCTIONS ################################################################

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

def print_ruptures_results(row_df):
    print('----- Results for optimisation approach \n using '+ 
          row_df['Cost_function'][0] +' with '+ row_df['Search_direction'][0] +
          ' \n \n'+ str(row_df['CPs'][0]) +
          '\n\n K: ' + str(row_df['K'][0]) +
          '\n AE: ' + str(row_df['AE'][0]) +
          '\n Meantime: ' + str(row_df['MeanTime'][0]) +
          '\n Precision: ' + str(row_df['Percision'][0]) +
          '\n Recall: ' + str(row_df['Recall'][0]) +
          '\n F1: ' + str(row_df['F1'][0]) +
          '\n Randindex: ' + str(row_df['RI'][0]) +
          '\n ------------------------------------------------------ \n\n')                

def print_bayes_results(CPs_true, CPs):
    import ruptures.metrics as rpt_met
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
    
    print('----- Results for Bayesian approach'
                  #' ----- \n Using: '+ prior_name +'prior and '+ likelihood_name +
                  #' likelihood function \n PAA window: '+ str(PAA_size)+
                  '\n \n' +  str(CPs) +
                  '\n \n K: ' + str(K) +
                  '\n AE: '+ str(AE) +
                  '\n Meantime: ' + str(MT) +
                  '\n Precision: '+ str(Precision[0])+
                  '\n Recall: ' +str(Precision[1])+
                  '\n F1: ' + str(F1) +
                  '\n Randindex: ' + str( RI) +
                  '\n ------------------------------------------- \n \n')

### MAIN SCRIPT ##############################################################

data_tag = 'P2'             # P1, P2, P3, P4, P5, P6, PRONTO

# Settings                  # SELECTIONS
search_direction = 'WIN'    # WIN or PELT
cost_functions = ['ridge', 'lasso', 'ar', 'l1', 'l2']   # l1, l2, ar, normal, LinReg, ridge, lasso
selection_metric = 'F1'     # 
maximise = True             	# If false, minimise

bayesian = True

# Esthetic settings
ruptures_y = -0.1
ruptures_space = -0.1

bayes_y = -0.8
bayes_space = -0.1

print_results = True       # True or False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Plot the data and true change points

directory = 'Datasets/'+data_tag+'/Results'

data, CPs_true = read_dataset_folder(data_tag)
if data.shape[1]!=1:
    data = data[data.columns[0]]

plot_shifting_backgorund(data, CPs_true)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Ruptures results

print('Dataset '+ data_tag +'\n - - - - - - - - - - - - - - - -')
for cost in cost_functions:
    for file in os.listdir(directory):
        if file.startswith(cost+'_'+search_direction):
            
            # Read file and filter search direction
            df = pd.read_csv(directory+'/'+file)
            
            # Maximise or minimise
            if maximise:
                row = df[selection_metric].argmax()
            else:
                row = df[selection_metric].argmin()
            
            if print_results:
                print_ruptures_results(df.iloc[[row]])
            
            cps = df['CPs'][row].split()[1:-1]
            
            # Plot the predictions and update new y position
            plt.plot(np.array(cps).astype(int), np.ones(len(cps))*ruptures_y , 'o', label = cost)
            ruptures_y  += ruptures_space
del cost, file, row, cps, df        

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Bayesian results

if bayesian:
    for file in os.listdir(directory):
        if file.startswith('Bayes_CPs'):
            df = pd.read_csv(directory+'/'+file)
            cps = df['0'].values
            
            # Plot the predictions and update new y position
            lab = 'Bayes '+file[-12:-4]
            plt.plot(cps, np.ones(len(cps))*bayes_y , 'X', label = lab )
            bayes_y  += bayes_space
            
            if print_results:
                print_bayes_results(CPs_true, cps)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Add title etc
plt.title('Predictions on ' + data_tag, fontsize = 14)
plt.legend(loc = 'upper left', fontsize = 14)


# Read Bayes
