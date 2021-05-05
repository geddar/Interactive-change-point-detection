# Interactive-change-point-detection
A repository for the work produced during my maters thesis [1] on the optimisation and Bayesian approaches. 

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

## Licence


### Ruptures
Copyright (c) 2017, ENS Paris-Saclay, CNRS
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

