%% Conformal inference
% Method: run conformal inference on a data set with Lasso
% runs lasso for all points in ytrial. 
%% Method
function [yconf,ms] = conformalLasso(X,Y,xnew,alpha,ytrial,lambda)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% ytrial    trail set for y

% prepare for fitting
n = length(ytrial);
addpath(genpath(pwd));

X_withnew = [X;xnew];
Pi_trial = zeros(1,n);
[m,p] = size(X);
modelsizes = zeros(1,n);
yconfidx = [];

h = waitbar(0,'Please wait...');
for i = 1:n
    y = ytrial(i);
    beta = lasso(X,Y,'Lambda',lambda/m,'Standardize',0,'RelTol',1E-8);
    Resid = abs(X_withnew*beta - [Y;y]);
    Pi_trial = sum(Resid<=Resid(end))/(m+1);
    if Pi_trial<=ceil((1-alpha)*(m+1))/(m+1)
        yconfidx = [yconfidx i];
    end
    waitbar(i/n,h,...
        sprintf('Current model size %d. Number of models computed %d'...
        ,length(find(beta)),i))
    modelsizes(i)=length(find(beta));
end
close(h) 
yconf = ytrial(yconfidx);
ms = mean(modelsizes);
    
    
