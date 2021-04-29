# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:58:50 2021
A script to explore datafiles (not genreate, load from directory)

@author: DEREGED1
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
#from scipy.signal import find_peaks

#import bayesian_changepoint_detection.offline_changepoint_detection as offcd
#from functools import partial

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
    return df, data, CPs, Bayes_pcp, CPs_bayes

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
problem_tag = 'P3'

df, data, CPs_true, Bayes_pcp, CPs_bayes = read_directory_files(r'C:/Users/DEREGED1/.spyder-py3/metrics_data/'+problem_tag)
df['min_metric'] = df['AE']*df['MeanTime']*df['RI']

# Divide into PELT and WIN
df_PELT = df[df['Search_direction']== 'PELT']
df_WIN = df[df['Search_direction']=='WIN']

#cp_bayes, _ = find_peaks(Bayes_pcp.values.reshape(1,-1)[0], height = 0.2, distance = 10)
#cp_bayes = np.concatenate([cp_bayes, [data.shape[0]]])

#CPs_bayes = pd.DataFrame(cp_bayes)
#CPs_bayes.to_csv(r'C:/Users/DEREGED1/.spyder-py3/metrics_data/'+problem_tag+'/CPs_bayes.dat')

# % PRINT METRICS FOR SOME PREDICTIONS
print('BAYESIAN PREDICTIONS')
print(problem_tag)
print_metrics(CPs_true, CPs_bayes)

# %% Plot data and true datapoints

plot_shifting_backgorund(data, CPs_true.values)
plt.title('Changing variance (PELT)', fontsize=16)

cp2 = [ 185,  370,  450,  640,  930, 1005]

plt.plot(cp2, np.ones(len(cp2))*-5,  'o', label = "Normal")

cp1 = [ 115,  275,  375,  520,  635,  740,  900, 1005]
plt.plot(cp1, np.ones(len(cp1))*-5.6,  'o', label = "LinReg")

cp3 = [ 110,  210,  310,  470,  575,  675,  800,  900, 1005]
plt.plot( cp3, np.ones(len(cp3))*-6,  'o', label = "Ridge")

plt.plot( CPs_bayes, np.ones(len(CPs_bayes))*-7, 'X', label = 'Bayesian' )

plt.legend(loc='upper left', fontsize=16)

# %% PLOT COSTS

#list_of_sets = [l1_WIN_changeing_var, l2_WIN_changeing_var, ar_WIN_changeing_var, clinear_WIN_changeing_var, normal_WIN_changeing_var, m_WIN_changeing_var]#, l2_total, ar_total, m_penalty_100, clinear_penalty, normal_total]

for metric in ['K', 'AE','MeanTime', 'Percision', 'Recall', 'F1']:    
    plot_all_costs(metric, df_WIN , min_val= 0, max_val=40)


# %% PLOT CPS

# Piecewise constant
CP_est = [40, 305, 340, 605, 640, 900]

# Lin_data
#CP_est = [105, 190, 300, 405, 505, 600] # AR
#CP_est_N = [105, 195, 300, 480, 600] # Normal

# Variance data
#CP_est_N = [185, 370, 685, 760, 850, 930, 1005] # Normal pen = 0 , MT = 22
#CP_est = 	[ 185,  370,  930, 1005] # Norma, pen = 1:49  MT = 0.6666

# AR data
#CP_est = [150, 305, 450, 605, 750, 900] # AR

# Exp_lin
#CP_est_N =	[200, 300, 500, 600, 800, 900] # AR, pen 0, MT = 0
#CP_est = [200, 340, 500, 630, 800, 900] # Normal, pen = 0, MT = 14

# Oscil data
#CP_est_N =	[ 50, 150, 220, 325, 400, 500, 570, 700] # Normal, pen 0,  MT = 27.14,
#CP_est =	[ 60, 130, 230, 310, 410, 485, 585, 700] # Clinear, pen 0, MT = 30.71,

# PELT stuff
CP_est=[40,305,340, 605, 640,900]

#rpt.display(data, CPs_true.values, CP_est_N)

#predicted_cp_RUPTURES_N = np.zeros(data.shape)
#predicted_cp_RUPTURES_N[np.array(CP_est_N)-1]=1

predicted_cp_RUPTURES = np.zeros(data.shape)
predicted_cp_RUPTURES[np.array(CP_est)-1]=1

# %%

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



# %% Plot both approache's results

fig, axs = plt.subplots(2, 1, sharex=(True), figsize = [18,10])

axs[0].plot(data)
axs[0].plot(CPs_true.values.reshape(1,-1)[0] - 1, data.values.reshape(1,-1)[0][CPs_true.values.reshape(1,-1)[0] - 1], 'o')
axs[0].set_title('Simulated data', fontsize=20)

#axs[1].plot(predicted_cp_RUPTURES_N,'r', label = 'Normal')
#axs[1].plot(predicted_cp_RUPTURES, label = 'L1')
#axs[1].set_title('Optimisation approach', fontsize=16)
#axs[1].legend(loc='upper left', fontsize=16)

axs[1].plot(Bayes_pcp)
axs[1].set_title('Bayesian approach', fontsize=16)

fig.tight_layout()
plt.show()

# %% 
rpt.base.BaseCost()
