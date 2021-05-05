# Interactive-change-point-detection
A repository for the work produced during my maters thesis [1] on the optimisation and Bayesian approaches. 

The work is based on development from other contributers, where _Ruptures_ [2] forms a foundation for the optimisation approach. The repository 
_bayesian_changepoint_detection_ by Johannes Kulick https://github.com/hildensia/bayesian_changepoint_detection (besed on work by Fearnhead [3]) forms the foundation for the Bayesian approach.

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
[1] 
[2]
[3]

## Licence

### Bayesian Changepoint Detection
Copyright (c) 2014 Johannes Kulick

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

