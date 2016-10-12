%% Conformal inference
% Method: run conformal inference on a data set,
% Check if the new point is in known support, if yes, subgradient method
% if no, run full lasso
%% Method
function [yconf,modelsize,supportcounter,triallen] = conformalRidge(X,Y,xnew,alpha,ytrial,lambdain)
% X, Y      input data, in format of matrix
% xnew      new point of x
% alpha     level
% stepsize  stepsize of searching for upper and lower bound of interval
% call it with lambda to use the fixed lambda method
% call it without lambda to use the cv lambda method

% prepare for fitting
addpath(genpath(pwd));
X_withnew = [X;xnew];
[m,p] = size(X);

% Tune GLMNET
options = glmnetSet();
options.standardize = false;        % original X
options.intr = false;               % no intersection
options.standardize_resp = false;   % original Y
options.alpha = 1.0;                % Lasso (no L2 norm penalty)
options.thresh = 1E-12;
options.nlambda = 1;
options.lambda = lambdain/m;

%% Fit the known data. 
% this is the condition of the new pair being outlier
% use this condition to truncate the trial set
tau=1;

t = X_withnew*((X_withnew'*X_withnew+tau*eye(p))\X_withnew');

A = t*[Y;0] - [Y;0];
B = t(:,end);
B(end) = B(end)-1;

root1 = (A(m+1)-A(1:m))./(B(1:m)-B(m+1));
root2 = (-A(1:m)-A(m+1))./(B(m+1)+B(1:m));
Cleft = sort(min(root1,root2));
Cright = sort(max(root1,root2));

tmin = prctile(Cleft, 100*alpha/2);
tmax = prctile(Cright,100 - 100*alpha/2);


modelsize = 0;
supportcounter = 1;
triallen = 0;
yconf  = [tmin,tmax];

end