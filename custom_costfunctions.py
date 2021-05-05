# -*- coding: utf-8 -*-
"""
File containing custom cost function for the ruptures package.

Implemented cost functions:
    - Linear regression
    - Ridge regression
    - Lasso regression

@author: Rebecca Gedda
"""
import numpy as np

from ruptures.base import BaseCost
from numpy.linalg import lstsq
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

reg_const = 1

##############################################################################
#                    LINEAR REGRESSION
##############################################################################

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


##############################################################################
#                    RIDGE REGRESSION
##############################################################################

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


##############################################################################
#                    LASSO REGRESSION
##############################################################################

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