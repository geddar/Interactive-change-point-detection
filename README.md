# Interactive-change-point-detection
A repository for user friendly change point detection using an optimisation approach and Bayesian statistics.
This work was produced during my maters thesis [1] where a comparison between the twp appoaches was studied.

The work is based on development from other contributers, where _Ruptures_ by Truonga et al. (based on work [2] forms a foundation for the optimisation approach. The repository 
_bayesian_changepoint_detection_ by Johannes Kulick (besed on work by Fearnhead [3]) forms the foundation for the Bayesian approach.
Respective repositories are found here:
- https://github.com/deepcharles/ruptures
- https://github.com/hildensia/bayesian_changepoint_detection

This repository formas a user friendly environment to make offline change point detection using both approaches.

## Prediction procedure
To add a dataset, one creates a folder in the Datasets folder with an appropriate name tag. 
The folder should contain:
- A data file (csv or dat) with any name
- CPs.dat, A file containing true change points
- Results: an empty folder to store results

Predictions are made using respective files:
- _calculate_predictions_Bayes.py_
- _calculate_predictions_ruptures.py_

Files are run serately, and parameters for the predictions are specified in the file.

Results are saved in the problem specified folder, in a sub-folder. 
Visualisation of the results can be made using _visualise_results.py_.

## Requirements
Requires installation of:
- ruptures

## References
[1] Rebecca Gedda, _Interactive Change Point Detection Approahces in Time-Series_ (2021)

[2] Charles Truonga  et al.,  _Selective  review  of  offline  change  point  detection methods_ (2020)

[3] Paul Fearnhead, _Exact and Efficient Bayesian Inference for Multiple Changepoint problems_ (2006)
