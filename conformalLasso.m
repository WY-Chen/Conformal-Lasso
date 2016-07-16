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
[m,p] = size(X);
modelsizes = zeros(1,n);
yconfidx = [];
%% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;
options.nlambda = 1;
options.lambda = lambda/m;

h = waitbar(0,'Please wait...');
for i = 1:n
    y = ytrial(i);
    beta = glmnetCoef(glmnet(X_withnew,[Y;y],[],options));
    beta = beta(2:p+1);
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
    
    
