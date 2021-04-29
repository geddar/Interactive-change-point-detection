# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:59:37 2021

@author: DEREGED1
"""
# IMPORTS ####################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from ruptures import metrics as rpt_met
import os
from sklearn import preprocessing

# % Custom cost functions
#from math import log
from ruptures.base import BaseCost
from numpy.linalg import lstsq
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# %% COST FUNCTIONS ##########################################################
class CostLinReg(BaseCost):

    """Custom cost for exponential signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 100
    
    def fit(self, signal):
        """Set the internal parameter."""
        #assert signal.ndim > 1, "Not enough dimensions"
        
        self.signal = signal.values.reshape(-1,1)
        self.covar = np.array(signal.index).reshape(-1,1)
        self.max = np.max(signal)
        self.min = np.min(signal)
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        
        y = self.signal[start:end]
        X = self.covar[start:end]
        
        y, X = self.signal[start:end], self.covar[start:end]
        _, residual, _, _ = lstsq(X, y, rcond=None)
        return residual.sum()

class CostRidge(BaseCost):

    """Custom cost for exponential signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 100
    
    def fit(self, signal):
        """Set the internal parameter."""
        #assert signal.ndim > 1, "Not enough dimensions"
        
        self.signal = signal.values.reshape(-1,1)
        self.covar = np.array(signal.index).reshape(-1,1)
        self.max = np.max(signal)
        self.min = np.min(signal)
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        
        y = self.signal[start:end]
        X = self.covar[start:end]
        #clf = Ridge(alpha= np.abs(self.max - self.min) )
        clf = Ridge(alpha= 1 )
        clf.fit(X,y)
        residual = sum( np.abs( y.reshape(1,-1)[0] - clf.predict(X).reshape(1,-1)[0]) )
        return residual

class CostLasso(BaseCost):

    """Custom cost for exponential signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2
    
    def fit(self, signal):
        """Set the internal parameter."""
        #assert signal.ndim > 1, "Not enough dimensions"
        
        self.signal = signal.values.reshape(-1,1)
        self.covar = np.array(signal.index).reshape(-1,1)
        self.max = np.max(signal)
        self.min = np.min(signal)
        self.length = len(signal)
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        
        y = self.signal[start:end]
        X = self.covar[start:end]
        clf = Lasso(alpha = 1)
        clf.fit(X,y)
        
        residual = sum( np.abs( y.reshape(1,-1)[0] - clf.predict(X).reshape(1,-1)[0]))
        return residual



# %% FUNCTIONS ###############################################################

def calculate_predictions(data, name_tag, search_dir, cost_f, c_start, c_end, c_step, CPs_true):
    penalty_results = pd.DataFrame()
    # Save K_true
    K_true = len(CPs_true)
    
    # Name the data
    name = name_tag
    
    window_width = 100

    # Select search direction: ["Window", "PELT"]
    search_direction = search_dir

    #Cost fuctions: ["l1", "l2", "ar", "clinear", "normal", "mahalanobis"]
    #for cost_function in [cost_f]:
    for penalty in range(c_start, c_end, c_step):
        CPs_Ruptures = []
        print(penalty)
        
            
        # Select search direction
        if search_direction == "WIN":
            if cost == 'LinReg':
                algo = rpt.Window(width = window_width, custom_cost=CostLinReg()).fit(data)
            elif cost == 'ridge':
                algo = rpt.Window(width = window_width, custom_cost=CostRidge()).fit(data)
            elif cost == 'lasso':
                algo = rpt.Window(width = window_width, custom_cost=CostLasso()).fit(data)
            else:
                algo = rpt.Window(width = window_width, model=cost_f).fit(data.values)
        elif search_direction == "PELT":
            if cost == 'LinReg':
                algo = rpt.Pelt(custom_cost=CostLinReg()).fit(data)
            elif cost == 'ridge':
                algo = rpt.Pelt(custom_cost=CostRidge()).fit(data)
            elif cost == 'lasso':
                algo = rpt.Pelt(custom_cost=CostLasso()).fit(data)
            else:
                algo = rpt.Pelt(model=cost_f).fit(data.values)
        else: 
            print("Give another search direction (Window or PELT) ")
        #    break
    
        result = algo.predict(pen = penalty)
        
        CPs_Ruptures.append(result)
        
        #CPs_Ruptures = np.concatenate(CPs_Ruptures, axis=0 )
        CPs_Ruptures = np.append(CPs_Ruptures, len(data))
        #CPs_Ruptures.append(len(data))
        CPs_Ruptures = np.unique(CPs_Ruptures)
        K_Ruptures = len(CPs_Ruptures)
                
        
        # % METRICS
        # Annotation error
        AE_Ruptures = np.abs(K_Ruptures - K_true)

        # Meantime
        MT_Ruptures = rpt_met.meantime(CPs_true, CPs_Ruptures)

        # RandIndex
        RandIndex_Ruptures = rpt_met.randindex(CPs_Ruptures, CPs_true)
        
        # Precision/Recall (Note margin can be changed)
        margin = np.ceil(data.shape[0]*0.05)
        #margin = 10
        Precision_Ruptures = rpt_met.precision_recall(CPs_true, CPs_Ruptures, margin = margin)
        
        # F1-score
        #F1 = 2* (Perc * recall)/(Perc + recall)
        if Precision_Ruptures[0] + Precision_Ruptures[1] != 0:
            F1_Ruptures = (2*Precision_Ruptures[0]*Precision_Ruptures[1])/(Precision_Ruptures[0] + Precision_Ruptures[1])
        else:
            F1_Ruptures = 0
        
        #penalty_results = penalty_results.append([[search_direction, cost_function, penalty, K_Ruptures, AE_Ruptures, H_Ruptures, MT_Ruptures, #RandIndex_Ruptures, 
        #                                           Precision_Ruptures[0], Precision_Ruptures[1], F1_Ruptures, CPs_Ruptures]], ignore_index=True)
        penalty_results = penalty_results.append([[search_direction, cost_f, penalty, K_Ruptures, AE_Ruptures, MT_Ruptures, #RandIndex_Ruptures, 
                                                   Precision_Ruptures[0], Precision_Ruptures[1], F1_Ruptures, RandIndex_Ruptures, CPs_Ruptures]], ignore_index=True)
    
    cols = ["Search_direction","Cost_function","Penalty", "K", "AE", "MeanTime", #"RandIndex", 
            "Percision", "Recall", "F1", "RI", "CPs"]
    penalty_results.columns = cols
    penalty_results.to_csv('metrics_data/'+name+'/'+cost_f+'_'+search_dir+'_'+name+'.csv')
    #penalty_results.to_csv(cost_f+'_'+search_dir+'_'+name+'.csv')
    return penalty_results

def read_data_files(path):

    directory = path
    
    for filename in os.listdir(directory):
        if filename.endswith(".dat"):
            if filename == 'CPs.dat':
                CPs = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
            elif filename == 'Bayes_flat_prior.dat':
                pcp = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
            else:
                #data = pd.read_csv(filename)
                data = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
    return data, CPs

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
        else:
            #data = pd.read_csv(filename)
            data = pd.read_csv(os.path.join(directory, filename), delimiter=",", index_col=0)
    return df, data, CPs, Bayes_pcp

# %% RUN TEST ################################################################
problem_tag = 'P6'

# Indicate which folders
for data_tag in [problem_tag]: #['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    
    print('Starting problem: ' + data_tag)
    data, CPs_true = read_data_files('metrics_data/'+ data_tag)
    
    
    # NORMALISED SIGNALS
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(data.values)
    norm_df = pd.DataFrame(x_scaled)
    norm_df.columns = data.columns
    
    plt.figure(figsize=(20,4))
    plt.plot(data)
    plt.plot(CPs_true.values.reshape(1,-1)[0] - 1, data.values.reshape(1,-1)[0][CPs_true.values.reshape(1,-1)[0] - 1], 'o')
    
    # Indicate search directions ['WIN', 'PELT']
    for search_direction in ['PELT']:
        if search_direction == 'WIN':
            start = 0
            end = 2
            jump = 1
        else:
            start = 0
            end = 20
            jump = 5
         
        # Indicate cost functions ["l1", "l2", "ar", "clinear", "normal", "mahalanobis", 'LinReg', 'ridge', 'lasso']
        for cost in  ['ridge', "lasso" ]:
            print(search_direction + '    ' + cost)
            some_df = calculate_predictions(data, data_tag, search_direction, cost, start, end, jump, CPs_true.values.reshape(1,-1)[0])
    
    print(data_tag + ' DONE!')
    
 # % Read all files in a directory

df, data, CPs_true, Bayes_pcp = read_directory_files(r'C:/Users/DEREGED1/.spyder-py3/metrics_data/'+problem_tag)
df['min_metric'] = df['AE']*df['MeanTime']*df['RI']

# Divide into PELT and WIN
df_PELT = df[df['Search_direction']== 'PELT']
df_WIN = df[df['Search_direction']=='WIN']
