# Conformal Prediction with Lasso-like methods

Conformal prediction using Lasso support. 

## Problem in concern

Running conformal prediction with Lasso is extremely slow because it requires fitting a full lasso for every trial value. 

## Possible solution and General Steps

Since Lasso is cheap to compute if given the supports and signs of the model, we hope the trial value does not change the supports and signs, and thus reduce the incidents of computing full lasso.  In general, the method bases on the two modes:

1. __Fitting full lasso__: fit full lasso on some collection of data, and get the polyhedron conditions for support and signs derived from KKT conditions. And do conformal prediction.
2. __Fitting lasso using known subgradient__: Check if the new trial pair lies within the polyhedron of known support and signs, if not, fit full lasso, if yes, use the simplified computation. And do conformal prediction.

# Methods

## Full Lasso 

_(conformalLasso.m)_

Fit full lasso for each trial value. 

__Advantage__: this is the original definition of conformal prediction. It only relies on the assumption of weak exchangeability (thus, assumption-free).

__Problem__: Slow, as stated above. 

## LASSO All Support

_(conformalLassoAllSupp.m)_

1. Traverse the trial set, for each new trial yi, run full Lasso if the combined data  (X,xnew) cross (Y,yi) is not in the previous support, and record this new support; run subgradient Lasso if the  (X,xnew) cross (Y,yi) is in previous support.  
2. Construct confidence Interval. 

__Advantage__: This method works well with different generative simulation data types. There is a similar method [Sparse Conformal Predictors _by Mohamed Hebiri_](http://arxiv.org/abs/0902.1970) which does this to the whole lasso path. Effectively, this method yields the same output as running lasso at each point. It does not depend on any additional assumption. 

__Problems__: Sometimes can be slow if the data is heavily depended on the new data pair, which results in constant support/sign changes. In the worst case, it is equivalent to fitting full lasso at each point. In these cases it can be even slightly slower then the full lasso method, for it computes in addition the supports. However, this does not happen much, even when the data is extremely noisy. And on other occasions, this method saves much time. 

__Notice__: I also implemented a cross-validation version that recalculates lambda each time the method fits a full lasso. The CV version sometimes gives a sharper interval, but takes considerably more time. Thus should never be used. 

## LTS-lasso All Support

_(conformalLTSLassoAllSupp.m)_

#### Conformal LTS-lasso:

1. __Get rid of outliers__: fit a lasso to the original _N_ data. Rank the residues to compute a interval for trial values that the trial pair is included in the selected data set _H_. Discard the rest trial values, because they does not satisfy stopping condition of C-steps.
2. __LTS-Lasso C-Steps__:
   1. __Initialization__: randomly draw three data and fit a lasso. Rank the residues, take the first _h_ many to run another lasso; rerun twice. Do this 50 times, take the simple majority of selected data set _H_ to run C-steps.
   2. __C-Steps__: Given a subset _H_ of _h_ pairs of data, fit a lasso and rank the residues. Pick the lowest _h_ data as new set _Hnew_. Repeat while _H_ and _Hnew_ are different, or until 20 steps. The rest of data are seen as outliers. 
3. __All Support__: similar to Lasso All support. Traverse the trial set, compute models by LTS-lasso on known data with trial pair: If the new trial pair is within known polyhedron of support and signs, apply the simplified computation on the selected set _H_; if not, fit full LTS-lasso with C-Steps. 

__Advantage__: If the data has a very sharp model, this method is significantly faster than lasso-all support, for we do not need to fit full lasso for trial values that are far off (which was ususally hard to compute using original lasso because it is vulnerable against outlier). Effectively, we compute lass full lasso. 

__Problem__: 

1. In addition to the randomness of model fit to the data, there is also randomness in data selection: when the trial pair is included in the selected set, a change of trial value may result in swap of two pairs from the known data in/out of the selected set. In this case, and in the case of new point not in support, the method has to refit with full lasso.  This problem leads to frequent support/signs change, that in a lot of case it computes even more full lasso than the previous method does. _Thus it compromises the advantage of this method._

2. Setting the proportion of data to truncate, _1-tau_, is tricky. The more we discard, the less accurate the method is, because we are effectively wasting _(1-tau)%_ of data. For this reason, the confidence interval is usually not sharp,that it gives longer intervals than the full lasso method. Intuitively, we want _tau_ to be as large as possible, which results in the next method.

   __So it should never be used.__

__Notice__: The CV version runs even slower for most of the time, but gives the shortest (hence also the sharpest) confidence interval of all the working methods.  However, when the problem stated in the "problem" section occurs, the CV method takes forever, thus should never be used. 

Source: [LTS-lasso](https://arxiv.org/pdf/1304.4773.pdf)

## LOO(Leave One Out)-Lasso All Support

_(conformalLOO.m)_

This method is exactly the previous method when setting _h=m-1_, i.e., only discarding one point as outlier. However, both steps are easier to compute in this method. 

1. __Get rid of outlier__: fit lasso to known data, this is the same as the condition when the new pair is classified as outlier in C-Step. The prediction value at xnew plus/minus the largest residue is the interval for trial values that the pair is excluded in the selected data set as outliers. Discard these trial values.
2. __Fit lasso for each point__: Start with mode 1, do the following two modes of computation:
   1. __Mode 1__: for a new trial pair, apply initialization and C-steps described in previous method, get a LTS-Lasso fit. Do conformal prediction. Check if the next trial value is within the polyhedron, if yes, switch to mode 2. If not, continue with mode 1. 
   2. __Mode 2__: use the known support and signs to refit the data, rank the residues to check if the outlier is also the same.  __(a)__. If yes, proceed with mode 2 on the next trial value until the next one is not in known support. __(b)__ If not, rerun with mode 1.

__Advantage__: This method is the fastest one. It also gives a sharpest confidence interval, because we are using more data than the LTS method, and less affected by outliers than LassoAllSupport. For less noisy linear data (In setting A), this method is usually at least 3 times as fast as LassoAllSupp, and twice as fast as LTSAllSupport. When encountering data that results in frequent support changing (as described in the "problem" session of previous methods), this method is more stable, because the selected data set usually stays the same. And when the support changes from point to point, this method is at least as slow as LassoAllSupp. 

__Problem__: Not found yet. Testing.

__Notice__: theoretically, this method should totally replace the LTS-lasso, because we should believe that there are very few, if not none, real outliers in the data. 

-------

# Examples:

## Setting A (linear, classical)

_X_ is 200*2000 matrix, each drawn from iid normal (0,1). 

_beta_ is 2000*1 vector, _beta_ = [2,2,2,2,2,0,0,...,0]

_Y=beta*X+epsilon_ where _epsilon_ is iid normal (0,1)

## Setting C (linear, heteroskedastic, heavy-tailed, correlated features)

_X_ is 200*2000. Each _X{i,j}_ is equally possibly drawn from normal (0,1), pearson system random (0,1,1,5) and bernoulli (0.5). Then each column is generated by a convex combination _X{j}=.4X{j}+.3X{j-1}+.2X{j-2}+.1X{j-3}_.

_beta_ is 2000*1 vector, _beta_ = [2,2,2,2,2,0,0,...,0]

_Y=beta*X+epsilon_ where _epsilon_ is iid t(2).

## World GDP (Notice: technically invalid because violated weak exchangeability)

_X_ is 30*35 matrix. The GDP data of 30 countries in the 35 years (1980-2014). 

_Y_ is 30*1 vector of GDP of year 2015. 

Testing data is the data pairs of the other 129 countries. 

## Bike (unknown model, could be linear with skewed error)

_X_ is 100*117 matrix. The visit to each station of 100 bikes. 

_Y_ is 100*1 vector of total duration of travel. 

Testing data is the data pairs of the other 839 bikes. 

------

# Simulation Study







------

# Appendix: Bad Methods

The following methods are discarded for either inefficiency or inaccuracy. Some are worth noticing, thus are listed below. _the code for these are in the bad method folder_.

### Lasso One Support

1. Run Lasso on known data (X,Y), use the fitted parameters to determine a polyhedron for support and signs. 
2. Solve for a range of observed value that would land in the polyhedron if paired with the new point. 
3. Run conformal prediction with subgradient lasso only on the trial values in this range and determine confidence interval.

__Coverage not guaranteed__ 

__Problem__: the supports might change very frequently, thus the interval we are examining is too short. 

### LTS-lasso One Support

1. Set alpha=0.9, i.e., using only 90% of the data. Use recursive C-steps that run lasso only on the data points with residual in lower 90% quantile until the chosen set does not change. 
2. Use the parameter from the C-steps to fit and do conformal prediction on the whole range. 

__Coverage not guaranteed__ 

__Problem__: The method is technically incorrect, for the support changes if the new data pair is in the chosen set, while here we treat all cases as if the new data pair is already outlier.

### LAD-Lasso methods

The computation of this method is basically LAD with additional p many data pairs for constraint. However, this runs very slowly per trial, and is not as good for shrinkage empirically.

Need to see whether computation can be simplified with known support.

Need to see whether computation can be simplified with known support.

__Too slow__
Source: [LAD-lasso](https://www.researchgate.net/publication/4724848_Robust_Regression_Shrinkage_and_Consistent_Variable_Selection_Through_the_LAD-Lasso?enrichId=rgreq-c8d87c27a1813dd29252d4abf613721e-XXX&enrichSource=Y292ZXJQYWdlOzQ3MjQ4NDg7QVM6OTg5OTE0ODQxNzg0NDJAMTQwMDYxMjgxNzU0NA%3D%3D&el=1_x_2)

### Elastic net One Support

Do Lasso one support method with elastic nets instead of lasso with the same Lars algorithm. This method does not show robustness against outlier, and the support is even narrower (because the L2 penalty helps choose many correlated features rather than one, as in lasso). 

__Too slow__

Source: [Elastic net](http://www.jmlr.org/proceedings/papers/v28/yang13e.pdf)

------

