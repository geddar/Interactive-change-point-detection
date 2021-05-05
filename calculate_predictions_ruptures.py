# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:08:16 2021

IN: Folder tag, Dataset and CPs
OUT: Created files with predictions in Results folder

Creates a file for each prediction algorithm applied to the data.

@author: Rebecca Gedda
"""
##############################################################################
#                    PREDICTIONS WITH RUPTURES
##############################################################################

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from ruptures import metrics as rpt_met
import os

# Custom cost functions
import custom_costfunctions

### FUNCTIONS ################################################################


def select_algorithm(data, search_direction, cost):
    
    
    
    if search_direction == "WIN":
        if cost == 'LinReg':
            algo = rpt.Window(width = window_width, custom_cost=custom_costfunctions.CostLinReg()).fit(data)
        elif cost == 'ridge':
            algo = rpt.Window(width = window_width, custom_cost=custom_costfunctions.CostRidge()).fit(data)
        elif cost == 'lasso':
            algo = rpt.Window(width = window_width, custom_cost=custom_costfunctions.CostLasso()).fit(data)
        else:
            algo = rpt.Window(width = window_width, model=cost).fit(data.values)
    elif search_direction == "PELT":
        if cost == 'LinReg':
            algo = rpt.Pelt(custom_cost=custom_costfunctions.CostLinReg()).fit(data)
        elif cost == 'ridge':
            algo = rpt.Pelt(custom_cost=custom_costfunctions.CostRidge()).fit(data)
        elif cost == 'lasso':
            algo = rpt.Pelt(custom_cost=custom_costfunctions.CostLasso()).fit(data)
        else:
            algo = rpt.Pelt(model=cost).fit(data.values)
    else: 
        print("Give another search direction (Window or PELT) ")
    
    return algo
    
def print_metrics(tag, sd, cost, pen, CPs, K, AE, MT, Prec, Recall, F1, RI):
    
    print('----- Results for '+ tag +
                  ' ----- \n Using: '+ sd +', '+ cost +
                  '\n Pently: '+ str(pen)+
                  '\n \n' +  str(CPs) +
                  '\n K: ' + str(K) +
                  '\n AE: '+ str(AE) +
                  '\n MT: ' + str(MT) +
                  '\n Precision: '+ str(Prec)+
                  '\n Recall: ' +str(Recall)+
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

def calculate_predictions(name_tag, search_dir, cost_f, c_start, c_end, c_step, print_bool):
    
    data, CPs_true = read_dataset_folder(name_tag)
    
    penalty_results = pd.DataFrame()
            
    for penalty in range(c_start, c_end, c_step):
        
        CPs = []
        for column in data.columns:
             # Select algorithm
             sub_data = pd.DataFrame(data[column])
             algorithm = select_algorithm(sub_data, search_dir, cost_f)
    
             result = algorithm.predict(pen = penalty)
        
             CPs.append(np.array(result))
        
        # Append the last index and remove duplicates
        CPs = np.concatenate(CPs)
        CPs = np.append(CPs, np.array(len(data)))
        
        CPs = np.unique(CPs)
        
        
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
        
        if print_bool == True:
            print_metrics(name_tag, search_dir, cost_f, penalty, CPs, K, AE, MT, Precision[0], Precision[1], F1, RI)
        else:
            print('----- Done with ' + name_tag + 
                  '----- \n Using: '+ search_dir +', '+ cost_f +
                  '\n Pently: '+ str(penalty) +'\n\n')
        
        penalty_results = penalty_results.append([[search_dir, cost_f, penalty, K, AE, MT, 
                                                   Precision[0], Precision[1], F1, RI, CPs]], ignore_index=True)
    
    cols = ["Search_direction","Cost_function","Penalty", "K", "AE", "MeanTime", 
            "Percision", "Recall", "F1", "RI", "CPs"]
    penalty_results.columns = cols
    penalty_results.to_csv('Datasets/'+name_tag+'/Results/'+cost_f+'_'+search_dir+'_'+name_tag+'.csv')
    return CPs, penalty

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

data_tag = 'P2'             # P1, P2, P3, P4, P5, P6, PRONTO

# Settings                  # SELECTIONS
search_direction = 'WIN'    # WIN, PELT
cost_function = 'ridge'        # l1, l2, ar, normal, LinReg, ridge, lasso
window_width = 100

penalty_start = 2
penalty_end = 4
penalty_jump = 1

print_results = True       # True or False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

predictions, penatly = calculate_predictions(data_tag, search_direction, cost_function, 
                      penalty_start, penalty_end, penalty_jump, 
                      print_results)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

data, CPs_true = read_dataset_folder(data_tag)
if data.shape[1]!=1:
    data = data[data.columns[0]]

plot_shifting_backgorund(data, CPs_true)
plt.plot(predictions, np.ones(len(predictions))*-0.1, 'X', label = cost_function)
plt.title('Predictions on ' + data_tag + ' using '+search_direction +' '+cost_function + ' (Penalty = '+str(penalty_end)+')', fontsize = 14)
#plt.title('Predictions on '+ data_tag, fontsize = 14)
plt.legend(loc = 'upper left', fontsize = 14)