%% Conformal inference
% Method: run conformal inference on a data set

%% Method
function [yconf,modelsize,sc] = conformal(X,Y,xnew,alpha,A,ytrial)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% A         A method that get parameters for X
% ytrial    trail set for y

% prepare for fitting
n = length(ytrial);
addpath(genpath(pwd));

X_withnew = [X;xnew];
Pi_trial = zeros(1,n);
[m,p] = size(X);
if strcmp(A,'linear')
    mtd=@(y,x)x*regress(y,x);
end

if strcmp(A,'lasso')
    mtd=@(y,x)cvglmnetPredict(cvglmnet(x,y),x);
end

for i = 1:n
    y = ytrial(i);
    Yfit = mtd([Y;y],X_withnew);
    Resid = abs(Yfit - [Y;y]);
    Pi_trial(i) = sum(Resid<=Resid(end))/(m+1);
end
yconf = ytrial((m+1)*Pi_trial<=ceil((1-alpha)*(m+1)));
modelsize=0;sc=0; % not implemented yet
    
    
