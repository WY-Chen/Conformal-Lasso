%% Conformal inference
% Method: run conformal inference on a data set with LAD-Lasso
% runs LAD-lasso for all points in ytrial. 
%% Method
function [yconf,s,ms] = conformalLADLasso(X,Y,xnew,alpha,ytrial)
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
h = waitbar(0,'Please wait...');
for i = 1:n
    y = ytrial(i);
    beta = LADlasso(X_withnew,[Y;y]);
    Yfit = X_withnew*beta;
    Resid = abs(Yfit - [Y;y]);
    Pi_trial(i) = sum(Resid<=Resid(end))/(m+1);
    waitbar(i/n,h,sprintf('%2.2f',i/n));
end
close(h) 
yconf = ytrial((m+1)*Pi_trial<=ceil((1-alpha)*(m+1)));
s=1; ms=-1;
    
    
