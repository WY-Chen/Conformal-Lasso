# Conformal Prediction with Lasso

Conformal prediction using Lasso support. 

## Problem in concern

Running conformal prediction with Lasso is extremely slow because it requires fitting a lasso for every trial value. 

## Possible solution

Since Lasso is cheap to compute if given the subgradient (supports and signs) of the model, we hope the trial value does not change the supports and signs of the fitted lasso, and thus simplify the computation. 

## General Steps

1. For each trial value, treat it as known value and determine if the data lies in the polyhedron of a certain known lasso support, using linear inequalities derived from KKT conditions. 
2. If the combined data lies in the polyhedron, solve lasso fit with known support and signs. Do conformal prediction.

# Methods

## One Support

1. Run Lasso on known data (X,Y), use the fitted parameters to determine a polyhedron for support and signs. 
2. Solve for a range of observed value that would land in the polyhedron if paired with the new point. 
3. Run conformal prediction with subgradient lasso only on the trial values in this range and determine confidence interval.

## All Support 

1. Traverse the trial set, for each new trial y, run full Lasso if the combined data  (X,xnew) cross (Y,yi) is not in the previous support, and record this new support; run subgradient Lasso if the  (X,xnew) cross (Y,yi) is in previous support.  
2. Construct confidence Interval. 

# Comparison

1. By far, __All Support is the best working method__ because it does not make any additional assumption and is also computationally easy.
2. One Support method is fast and works for the ideal setting of no noise, but no data is so ideal that the support doesn't change. 
3. Prediction One Support method is faster version of One Support, which suffers from the same problem of One Support.
4. Prediction Multiple Support method fixes the problem of changing support, is also fast, and has good coverage. However its advantage would be compromised if encounters supports which contains only one trial value, which leads to a stopping too early, and gives an interval smaller than required.
5. All Support method does not have any of the problems above, but it can be significantly slower than all of them for some condition when the support changes frequently. While the previous methods would terminate in a short time no matter what, this method could potentially run full lasso on every trial value, which is as slow as the original method of running each lasso. 

## Discussion

1. All Support method and Prediction Multiple Support method is essentially the same method but the former traverses the whole trial set and the latter only 'grows' from the prediction point. By this nature, the former tends to give larger coverage and the latter tends to give smaller coverage. 
2. The next step of the study is to either give a better stopping condition to Prediction Multiple Support method or give a condition to abandon computation for All Support method. 

## Updates

- 16.6.21: First commit. Implemented conformal method with lasso, computation of support polyhedron, and Multiple Support method that does conformal prediction on different supports until the last one has no more valid point.
- 16.6.23: Implemented AllSupp method that traverse the whole trial set and run full lasso if not in previous support and subgradient lasso if in previous support. 
