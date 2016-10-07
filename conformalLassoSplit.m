function [yconf,mc,sc] = conformalLassoSplit(X,Y,xnew,alpha,~,lambdain)

% prepare for fitting
addpath(genpath(pwd));
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

% randomize two sets
I1 = randsample(1:m,floor(m/2));
I2 = setxor(1:m,I1);

beta = glmnetCoef(glmnet(X(I1,:),Y(I1),[],options));
beta = beta(2:p+1);
mu = xnew*beta;
fitI2 = X(I2,:)*beta;
resid = sort(abs(fitI2-Y(I2)));
d = resid(ceil((m/2+1)*(1-alpha)));

yconf = [mu-d,mu+d];
mc = length(find(beta));sc=1;