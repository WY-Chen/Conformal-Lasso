# Conformal Prediction with Lasso

Conformal prediction using Lasso support. 

## Problem in concern

Running conformal prediction with Lasso is extremely slow because it requires fitting a lasso for every trial value. 

## Possible solution

Since Lasso is cheap to compute if given the subgradient (supports and signs) of the model, we hope the trial value does not change the supports and signs of the fitted lasso, and thus simplify the computation. 

## General Steps

1. For each trial value, treat it as known value and determine if the data lies in the polyhedron of a certain known lasso support, using linear inequalities derived from KKT conditions. 
2. If the combined data lies in the polyhedron, solve lasso fit with known support and signs. Do conformal prediction.

## Updates

- 16.6.21: First commit. Implemented conformal method with lasso, computation of support polyhedron, and Multiple Support method that does conformal prediction on different supports until the last one has no more valid point.
