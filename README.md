# Interactive-change-point-detection
A repository for the work produced during my maters thesis [1] on the optimisation and Bayesian approaches. 

The work is based on development from other contributers, where _Ruptures_ [2] forms a foundation for the optimisation approach, and 
_bayesian_changepoint_detection_ [3] forms the foundation for the Bayesian approach.

This repository formas a user friendly environment to make offline change point detection using both approaches.

## Prediction procedure
To add a dataset, one creates a folder in the Datasets folder with an appropriate name tag. 
The folder should contain:
- A data file (csv or dat) with any name
- CPs.dat, A file containing true change points
- Results: an empty folder to store results

Predictions are made using respective files:
- calculate_predictions_Bayes.py
- calculate_predictions_ruptures.py
which are run respectively. Parameters for the predictions are specified in the file.

Results are saved in the problem specified folder, in a sub-folder. 
Visualisation of the results can be made using visualise_results.py.

## Requirements
Requires installation of:
- ruptures

## Licence

## references
[1] 
[2]
[3]
