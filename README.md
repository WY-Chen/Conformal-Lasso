# Conformal Prediction with Lasso-like methods

Conformal prediction using Lasso support. 

## Problem in concern

Running conformal prediction with Lasso is extremely slow because it requires fitting a lasso for every trial value. 

## Possible solution

Since Lasso is cheap to compute if given the subgradient (supports and signs) of the model, we hope the trial value does not change the supports and signs of the fitted lasso, and thus simplify the computation. 

## General Steps

1. For each trial value, treat it as known value and determine if the data lies in the polyhedron of a certain known lasso support, using linear inequalities derived from KKT conditions. 
2. If the combined data lies in the polyhedron, solve lasso fit with known support and signs. Do conformal prediction.

# Methods

## LASSO methods

### One Support

1. Run Lasso on known data (X,Y), use the fitted parameters to determine a polyhedron for support and signs. 
2. Solve for a range of observed value that would land in the polyhedron if paired with the new point. 
3. Run conformal prediction with subgradient lasso only on the trial values in this range and determine confidence interval.

__Coverage not guaranteed__ 

__Problem__: the supports might change very frequently, thus the interval we are examining is too short. 

### All Support

1. Traverse the trial set, for each new trial y, run full Lasso if the combined data  (X,xnew) cross (Y,yi) is not in the previous support, and record this new support; run subgradient Lasso if the  (X,xnew) cross (Y,yi) is in previous support.  
2. Construct confidence Interval. 

__Advantage__: This method works. In fact, there is a similar method [Sparse Conformal Predictors _by Mohamed Hebiri_](http://arxiv.org/abs/0902.1970) 

__Problems__: Sometimes can be really slow if the data is heavily depended on the new data pair, which results in constant support/sign changes. 

------

## LAD-Lasso methods

The computation of this method is basically LAD with additional p many data pairs for constraint. However, this runs very slowly per trial, and is not as good for shrinkage empirically.

Need to see whether computation can be simplified with known support.

Need to see whether computation can be simplified with known support.
Source: [LAD-lasso](https://www.researchgate.net/publication/4724848_Robust_Regression_Shrinkage_and_Consistent_Variable_Selection_Through_the_LAD-Lasso?enrichId=rgreq-c8d87c27a1813dd29252d4abf613721e-XXX&enrichSource=Y292ZXJQYWdlOzQ3MjQ4NDg7QVM6OTg5OTE0ODQxNzg0NDJAMTQwMDYxMjgxNzU0NA%3D%3D&el=1_x_2)

------

## (in construction) Lasso with Huber loss

This is not hard to implement using the CVX package.

Need to figure out how to decide the shape and how to decide if a new y is in the support.

Need to figure out a simplified way to compute parameter given support

Source: [Huber Loss](http://arxiv.org/pdf/math/0406470.pdf)

------

## LTS-Lasso methods

### LTS-lasso One Support

1. Set alpha=0.9, i.e., using only 90% of the data. Use recursive C-steps that run lasso only on the data points with residual in lower 90% quantile until the chosen set does not change. 
2. Use the parameter from the C-steps to fit and do conformal prediction on the whole range. 

__Coverage not guaranteed__ 

__Problem__: The method is technically incorrect, for the support changes if the new data pair is in the chosen set, while here we treat all cases as if the new data pair is already outlier.

### LTS-lasso All Support

Similar to Lasso All support. Traverse the trial set, compute a model by LTS-lasso, check if the new pair is in the chosen 'good' data set. 

1. If new pair is with weight 0 (classified as outlier in C-step), run conformal prediction with the given model. 
2. If new pair is with weight 1 (classified as 'good' data in C-step) and is in the range of the previous support/signs polyhedron, refit the lasso with the chosen data set and given support.
3. if new pair is with weight 1 (classified as 'good' data in C-step) and is out of the range of the previous polyhedron, rerun the C-steps with new pair. Get new polyhedron and new chosen set. 

__Advantage__: this is significantly faster than lasso-all support, for we do not need to fit full lasso for trial values that are far off (which was ususally hard to compute using original lasso because it is vulnerable against outlier). Effectively, we compute lass full lasso. Also since this method is robust to outliers, it gives a sharper sparse result. 

__Problem__: the current version of this algorithm lacks cross-validation or BIC method of choosing lambda (now it uses the alpha-rescaled lambda of a cv-full lasso fit as magic number). Another real problem is that we are wasting 10% of data. 

Source: [LTS-lasso](https://arxiv.org/pdf/1304.4773.pdf)

------

## Elastic Net methods

### Elastic net One Support

Do Lasso one support method with elastic nets instead of lasso with the same Lars algorithm. This method does not show robustness against outlier, and the support is even narrower (because the L2 penalty helps choose many correlated features rather than one, as in lasso). 

Source: [Elastic net](http://www.jmlr.org/proceedings/papers/v28/yang13e.pdf)

------