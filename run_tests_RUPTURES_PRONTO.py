# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:43:57 2021

File to make predictions on the PRONTO dataset. 
This file prints the obtained predictions and metric values,
and does not save them as seperate files. 

@author:Rebecca Gedda
"""

### IMPORTS ##################################################################
import matplotlib.pyplot as plt
import numpy as np
from ruptures.base import BaseCost
from numpy.linalg import lstsq
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import preprocessing

import ruptures as rpt
import ruptures.metrics as rpt_met

import pandas as pd
# %%
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

def print_metrics(CPs_true, CPs_predictions):
   
    print('Number of predictions (K): '+ str(CPs_predictions.shape[0]))
    
    # Annotation error
    AE = np.abs(CPs_true.shape[0] - CPs_predictions.shape[0])
    print('AE: '+ str(AE))

    # Meantime
    MT = rpt_met.meantime(CPs_true, CPs_predictions)
    print('Meanitme: '+ str(MT))
    
    # Precision/Recall (Note margin can be changed)
    #margin = 250
    margin = np.ceil(CPs_true[-1]*0.01)
    Precision = rpt_met.precision_recall(CPs_true, CPs_predictions, margin = margin)
    print('Precision: '+ str(Precision[0]))
    print('Recall: '+ str(Precision[1]))
    
    # F1-score
    #F1 = 2* (Perc * recall)/(Perc + recall)
    if Precision[0] + Precision[1] != 0:
        F1 = (2*Precision[0]*Precision[1])/(Precision[0] + Precision[1])
        print('F1: '+ str(F1))
    
    # RandIndex
    #RI = rpt_met.randindex(CPs_predictions, CPs_true)
    #print('RandIndex: '+ str(RI))

# %
### COST FUNCTIONS ##########################################################
reg_const = 1

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
        clf = Ridge(alpha=reg_const)
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
        #clf = Lasso(alpha = 10)
        clf = Lasso(alpha = reg_const)
        clf.fit(X,y)
        
        residual = sum( np.abs( y.reshape(1,-1)[0] - clf.predict(X).reshape(1,-1)[0]))
        return residual

# %%
### MAIN SCRIPT ##############################################################
# EXPERIMENT DAY 4: 12-09-2017
df = pd.read_csv('TD_0912/0912Testday4.csv')
CPs_true = pd.read_csv('TD_0912/CPs.dat')

# % Preparation of df
df.columns =  df.iloc[1].values
df = df.drop(0, axis = 0)
df = df.drop(1, axis = 0)
df = df.reset_index(drop = True)

# % Select relevant columns
df = df[['Air In1', 'Air In2', 'Water In1','Water In2']]
df = df.loc[:, ~df.columns.duplicated()]

# % convert to float from obj

for col in df.columns:
    df[col] = np.stack(df[col].values).astype(None)


# NORMALISED SIGNALS
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df.values)
norm_df = pd.DataFrame(x_scaled)
norm_df.columns = df.columns

# %
# % SETTING FOR TEXT #########################################################
cost = 'ridge'
penalty = 100
search_dir = 'WIN' # ['WIN', 'PELT']
window_width = 100
##############################################################################

print('Cost function: '+cost)
if cost == 'ridge' or cost == 'lasso':
    print('Reg const: '+str(reg_const))
#print('Penalty: '+ str(penalty))
print('Search direciton: '+ search_dir)
if search_dir == 'WIN':
    print('Window width (WIN): '+str(window_width))

#for penalty in [0.08, 0.06, 0.04, 0.02, 0.01, 0.009, 0.007, 0.005]:    
print('Penalty: '+ str(penalty))
predictions = []
for col in df.columns:
    print(col)
     # Chose data type
    if cost == 'LinReg' or  cost == 'ridge' or  cost == 'lasso' or cost == 'special':
        data = pd.DataFrame(norm_df[col])
    else:
        data = norm_df[col].values
    
    # Select search direction
    # WIN
    if search_dir == "WIN":
        if cost == 'LinReg':
            algo = rpt.Window(width = window_width, custom_cost=CostLinReg()).fit(data)
        elif cost == 'ridge':
            algo = rpt.Window(width = window_width, custom_cost=CostRidge()).fit(data)
        elif cost == 'lasso':
            algo = rpt.Window(width = window_width, custom_cost=CostLasso()).fit(data)
        else:
            algo = rpt.Window(width = window_width, model=cost).fit(data)
    
    # PELT
    elif search_dir == "PELT":
        if cost == 'LinReg':
            algo = rpt.Pelt(custom_cost=CostLinReg()).fit(data)
        elif cost == 'ridge':
            algo = rpt.Pelt(custom_cost=CostRidge()).fit(data)
        elif cost == 'lasso':
            algo = rpt.Pelt(custom_cost=CostLasso()).fit(data)
        else:
            algo = rpt.Pelt(model=cost).fit(data)
    else: 
        print("Give another search direction (Window or PELT) ")
    #    break
   
    result = algo.predict(pen = penalty)
    predictions.append(result)
predictions = np.concatenate(predictions)    
predictions = np.unique(predictions)
print(predictions)
print_metrics(CPs_true['0'].values, predictions)
print('-------------------------------------------------------------------')

#
# Plot the predictions
plot_shifting_backgorund(df['Air In2'], CPs_true['0'].values)
plt.plot(predictions, np.ones(len(predictions)), 'X', label = cost)
plt.title('Predictions using '+search_dir+' '+cost+ ' (Penalty = '+str(penalty)+')', fontsize = 14)
plt.title('PRONTO dataset, Normal cost function', fontsize = 14)
plt.legend(loc = 'upper left', fontsize = 14)
